# --------------------------------------------------------
# Based on LaBraM, EEGPT, CBraMod, BIOT, EEG_Image_decode, BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/935963004/LaBraM
# https://github.com/BINE022/EEGPT/tree/main/downstream
# https://github.com/wjq-learning/CBraMod
# https://github.com/ycq091044/BIOT
# https://github.com/ncclab-sustech/EEG_Image_decode
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

import os
import math
from typing import Iterable, Optional
import torch
from timm.utils import ModelEma
from timm.loss import LabelSmoothingCrossEntropy
import util.utils as utils
from util.utils import wandb_logger
import random
import matplotlib.pyplot as plt

def train_class_batch(model, samples, target, criterion):
    outputs = model(samples)
    loss = criterion(outputs, target)

    return loss, outputs

def train_one_epoch(args, model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    model_ema: Optional[ModelEma] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, ch_names=None):
    is_binary = (args.nb_classes == 1)
    
    # loss foundation
    if args.task_mod == 'Regression':
        criterion = torch.nn.MSELoss()
    elif args.nb_classes == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    model.train(True)
    if args.finetune_mod == 'linear':
        model.main_model.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    update_freq = args.update_freq
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        if args.norm_method == 'mv':
            samples = samples.float().to(device, non_blocking=True) * args.mv_norm_value
        else:
            samples = samples.float().to(device, non_blocking=True)
        
        targets = targets.to(device, non_blocking=True)
        if is_binary:
            targets = targets.float().unsqueeze(-1)
        else:
            targets = targets.int().long()

        # with torch.cuda.amp.autocast():
        loss, output = train_class_batch(
            model, samples, targets, criterion)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Warning: Loss is {}".format(loss_value))

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss /= update_freq
        grad_norm = loss_scaler(loss, optimizer, clip_grad=args.clip_grad,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(data_iter_step + 1) % update_freq == 0)
        
        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if args.task_mod == 'Classification':
            if is_binary:
                class_acc = utils.get_metrics(torch.sigmoid(output).detach().cpu().numpy(), targets.detach().cpu().numpy(), ["accuracy"], is_binary)["accuracy"]
            else:
                class_acc = (output.max(-1)[-1] == targets.squeeze()).float().mean()
            metric_logger.update(class_acc=class_acc)
        
        metric_logger.update(loss=loss_value)
        # metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        # metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            if args.task_mod == 'Classification':
                log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(args, data_loader, model, device, header='Test:', ch_names=None, metrics=['acc']):
    is_binary = (args.nb_classes == 1)
    
    if args.task_mod == 'Regression':
        criterion = torch.nn.MSELoss()
    elif is_binary:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    #header = 'Test:'

    # switch to evaluation mode
    model.eval()
    pred = []
    true = []
    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        EEG = batch[0]
        target = batch[-1]
        if args.norm_method == 'mv':
            EEG = EEG.float().to(device, non_blocking=True) * args.mv_norm_value
        else:
            EEG = EEG.float().to(device, non_blocking=True)
        
        target = target.to(device, non_blocking=True)
        if is_binary:
            target = target.float().unsqueeze(-1)
        else:
            target = target.int().long()
        
        # compute output
        # with torch.cuda.amp.autocast():
        output = model(EEG)
        loss = criterion(output, target)
        
        if is_binary and args.task_mod != 'Regression':
            output = torch.sigmoid(output).cpu()
        else:
            output = output.cpu()
        target = target.cpu()

        results = utils.get_metrics(output.numpy(), target.numpy(), metrics, is_binary)
        pred.append(output)
        true.append(target)

        batch_size = EEG.shape[0]
        metric_logger.update(loss=loss.item())
        for key, value in results.items():
            metric_logger.meters[key].update(value, n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))
    
    pred = torch.cat(pred, dim=0).numpy()
    true = torch.cat(true, dim=0).numpy()

    ret = utils.get_metrics(pred, true, metrics, is_binary, 0.5)
    ret['loss'] = metric_logger.loss.global_avg
    return ret

def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale

def get_model_output(model, samples, ch_names):
    outputs = model(samples)
    
    return outputs

def train_model(args, eeg_model, dataloader, optimizer, device, 
                img_features_all, config, loss_scaler, start_steps=None, 
                lr_schedule_values=None, wd_schedule_values=None, ch_names=None,
                num_training_steps_per_epoch=None, model_ema: Optional[ModelEma] = None):
    
    eeg_model.train()
    if args.finetune_mod == 'linear':
        eeg_model.main_model.eval()
    
    img_features_all = (img_features_all[::10]).to(device).float()

    if loss_scaler is None:
        eeg_model.zero_grad()
        eeg_model.micro_steps = 0
    else:
        optimizer.zero_grad()
    
    total_loss = 0
    correct = 0
    total = 0
    alpha=0.99
    features_list = []  # List to store features
    save_features= True
    for batch_idx, (eeg_data, labels, img_features) in enumerate(dataloader):
        step = batch_idx
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        optimizer.zero_grad()
        eeg_data = eeg_data.float().to(device, non_blocking=True)
        
        img_features = img_features.to(device).float()
        labels = labels.to(device)

        if loss_scaler is None:
            eeg_features = eeg_features.half()
            eeg_features = eeg_model(eeg_data).float()
        else:
            # with torch.cuda.amp.autocast():
            eeg_features = eeg_model(eeg_data).float()
        
        features_list.append(eeg_features)

        loss_scale = eeg_model.module.loss_scale if args.distributed else eeg_model.loss_scale
        img_loss = eeg_model.loss_func(eeg_features, img_features, loss_scale)
        loss = img_loss
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}.".format(loss_value))

        max_norm = args.clip_grad
        if loss_scaler is None:
            eeg_model.backward(loss)
            eeg_model.step()
            if model_ema is not None:
                model_ema.update(eeg_model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(eeg_model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=eeg_model.parameters(), create_graph=is_second_order,
                                    update_grad=True)
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(eeg_model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        total_loss += loss.item()
        logit_scale = loss_scale
        # Compute the corresponding logits
        logits_img = logit_scale * eeg_features @ img_features_all.T
        logits_single = logits_img
        predicted = torch.argmax(logits_single, dim=1) # (n_batch, ) in {0, 1, ..., n_cls-1}

        batch_size = predicted.shape[0]
        total += batch_size
        correct += (predicted == labels).sum().item()
        del eeg_data, eeg_features, img_features
    
    average_loss = total_loss / (batch_idx + 1)
    accuracy = correct / total

    return average_loss, accuracy, torch.cat(features_list, dim=0)

def evaluate_model(args, eeg_model, dataloader, device, img_features_all, k, config, ch_names):
    eeg_model.eval()

    img_features_all = img_features_all.to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    alpha = 0.99
    top5_correct = 0
    top5_correct_count = 0
    # Get all unique classes
    all_labels = set(range(img_features_all.size(0)))
    top5_acc = 0
    with torch.no_grad():
        for batch_idx, (eeg_data, labels, img_features) in enumerate(dataloader):
            eeg_data = eeg_data.float().to(device, non_blocking=True)
            
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            
            batch_size = eeg_data.size(0)
            eeg_features = eeg_model(eeg_data)
        
            logit_scale = eeg_model.loss_scale
            img_loss = eeg_model.loss_func(eeg_features, img_features, logit_scale)
            loss = img_loss
            
            total_loss += loss.item()
            
            for idx, label in enumerate(labels):
                # First, select k-1 classes excluding the correct class
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                selected_img_features = img_features_all[selected_classes]
                
                if k==200:
                    # Compute the corresponding logits
                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) in {0, 1, ..., n_cls-1}
                    if predicted_label == label.item():
                        correct += 1
                    _, top5_indices = torch.topk(logits_single, 5, largest =True)
                                                   
                    # Check if the true label is in the top-5 predictions
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                        top5_correct_count+=1                                
                    total += 1
                elif k == 50 or k == 100:
                    # For k=50 or 100, select k classes for evaluation
                    selected_classes = random.sample(possible_classes, k-1) + [label.item()]

                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    
                    predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    if predicted_label == label.item():
                        correct += 1
                    _, top5_indices = torch.topk(logits_single, 5, largest =True)
                                                   
                    # Check if the true label is in the top-5 predictions
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                        top5_correct_count+=1                                
                    total += 1
                elif k==2 or k==4 or k==10:
                    selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                    # Compute the corresponding logits
                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    # Get the predicted class
                    predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) in {0, 1, ..., n_cls-1}
                    if predicted_label == label.item():
                        correct += 1
                    total += 1
                else:
                    print("Error.")
            del eeg_data, eeg_features, img_features
    
    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total
    top5_acc = top5_correct_count / total

    return average_loss, accuracy, top5_acc

def main_train_loop(args, current_time, eeg_model, 
                    train_dataloader, test_dataloader, optimizer, 
                    device, img_features_train_all, img_features_test_all, 
                    config, loss_scaler, logger=None, lr_schedule_values=None, ch_names=None,
                    wd_schedule_values=None, num_training_steps_per_epoch=None, model_ema=None):
    logger = wandb_logger(config) if logger else None
    logger.watch(eeg_model,logger) 
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    v2_accs = []
    v4_accs = []
    v10_accs = []

    best_accuracy = 0.0
    best_epoch_info = {}
    results = []  # List to store results for each epoch
    for epoch in range(config.epochs):
        # Train the model
        start_steps=epoch * num_training_steps_per_epoch
        train_loss, train_accuracy, features_tensor = train_model(
            args, eeg_model, train_dataloader, optimizer, device, 
            img_features_train_all, config=config, loss_scaler=loss_scaler, 
            start_steps=start_steps, lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values, 
            ch_names=ch_names, num_training_steps_per_epoch=num_training_steps_per_epoch, model_ema=model_ema)
        if (epoch + 1) % args.save_ckpt_freq == 0:
        # Get the current time and format it as a string (e.g., '2024-01-17_15-30-00')
            save_dir = os.path.join(args.output_dir, 'saved_models')
            save_dir = f"{save_dir}/contrast/across/{config.model_name}_{current_time}"
            os.makedirs(save_dir, exist_ok=True)             
            file_path = f"{save_dir}/{epoch+1}.pth"
            torch.save(eeg_model.state_dict(), file_path)
            print(f"model saved in {file_path}!")
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        # Evaluate the model
        test_loss, test_accuracy, top5_acc = evaluate_model(args, eeg_model, test_dataloader, device, img_features_test_all, k=200, config=config, ch_names=ch_names)
        _, v2_acc, _ = evaluate_model(args, eeg_model, test_dataloader, device, img_features_test_all, k=2, config=config, ch_names=ch_names)
        _, v4_acc, _ = evaluate_model(args, eeg_model, test_dataloader, device, img_features_test_all, k=4, config=config, ch_names=ch_names)
        _, v10_acc, _ = evaluate_model(args, eeg_model, test_dataloader, device, img_features_test_all, k=10, config=config, ch_names=ch_names)
        _, v50_acc, v50_top5_acc = evaluate_model(args, eeg_model, test_dataloader, device, img_features_test_all, k=50, config=config, ch_names=ch_names)
        _, v100_acc, v100_top5_acc = evaluate_model(args, eeg_model, test_dataloader, device, img_features_test_all, k=100, config=config, ch_names=ch_names)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        v2_accs.append(v2_acc)
        v4_accs.append(v4_acc)
        v10_accs.append(v10_acc)
        
        # Append results for this epoch
        epoch_results = {
        "epoch": epoch + 1,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "v2_acc": v2_acc,
        "v4_acc": v4_acc,
        "v10_acc": v10_acc,
        "top5_acc":top5_acc,
        "v50_acc": v50_acc,
        "v100_acc": v100_acc,
        "v50_top5_acc":v50_top5_acc,
        "v100_top5_acc": v100_top5_acc
        }

        results.append(epoch_results)
        # If the test accuracy of the current epoch is the best, save the model and related information
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            # best_model_weights = model.state_dict().copy()
            
            best_epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "v2_acc":v2_acc,
                "v4_acc":v4_acc,
                "v10_acc":v10_acc
            }
        logger.log({
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy,
            "v2 Accuracy": v2_acc,
            "v4 Accuracy": v4_acc,
            "v10 Accuracy": v10_acc,
            "Epoch": epoch
        })

        print(f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Top5 Accuracy: {top5_acc:.4f}")
        print(f"Epoch {epoch + 1}/{config.epochs} - v2 Accuracy:{v2_acc} - v4 Accuracy:{v4_acc} - v10 Accuracy:{v10_acc} - v50 Accuracy:{v50_acc} - v100 Accuracy:{v100_acc}")
  
    # # Load the best model weights
    # model.load_state_dict(best_model_weights)

    # Create 5 subplots
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    # Loss curve
    axs[0, 0].plot(train_losses, label='Train Loss')
    axs[0, 0].plot(test_losses, label='Test Loss')
    axs[0, 0].legend()
    axs[0, 0].set_title("Loss Curve")

    # Overall accuracy curve
    axs[0, 1].plot(train_accuracies, label='Train Accuracy')
    axs[0, 1].plot(test_accuracies, label='Test Accuracy')
    axs[0, 1].legend()
    axs[0, 1].set_title("Accuracy Curve")

    # The following are the three new plots you added, assuming you've already calculated the corresponding accuracies
    # 2-class accuracy plot
    axs[1, 0].plot(v2_accs, label='2-class Accuracy')
    axs[1, 0].legend()
    axs[1, 0].set_title("2-Class Accuracy Curve")

    # 4-class accuracy plot
    axs[1, 1].plot(v4_accs, label='4-class Accuracy')
    axs[1, 1].legend()
    axs[1, 1].set_title("4-Class Accuracy Curve")

    # 10-class accuracy plot
    axs[2, 0].plot(v10_accs, label='10-class Accuracy')
    axs[2, 0].legend()
    axs[2, 0].set_title("10-Class Accuracy Curve")

    # Construct the string information for annotation
    info_text = (f"Best Model Info (from Epoch {best_epoch_info['epoch']}):\n"
                f"Train Loss: {best_epoch_info['train_loss']:.4f}\n"
                f"Train Accuracy: {best_epoch_info['train_accuracy']:.4f}\n"
                f"Test Loss: {best_epoch_info['test_loss']:.4f}\n"
                f"Test Accuracy: {best_epoch_info['test_accuracy']:.4f}\n"
                f"v2_acc:{best_epoch_info['v2_acc']:.4f}\n"
                f"v4_acc:{best_epoch_info['v4_acc']:.4f}\n"
                f"v10_acc:{best_epoch_info['v10_acc']:.4f}")

    axs[2, 1].axis('off')  
    axs[2, 1].text(0.5, 0.5, info_text, fontsize=10, ha='center', va='center', transform=axs[2, 1].transAxes)

    plt.tight_layout()

    # Add main title
    plt.suptitle('pos_img_text', fontsize=16, y=1.05)
    plt.savefig('pos_img_text')
    logger.finish()

    print(info_text)

    return results
