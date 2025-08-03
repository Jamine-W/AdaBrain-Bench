"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import torch
import torch.distributed.nn
from torch import distributed as dist, nn as nn
from torch.nn import functional as F

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
    image_features,
    text_features,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
    use_horovod=False,
):
    if use_horovod:
        assert hvd is not None, "Please install horovod"
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(
                    all_image_features.chunk(world_size, dim=0)
                )
                gathered_text_features = list(
                    all_text_features.chunk(world_size, dim=0)
                )
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(
                torch.distributed.nn.all_gather(image_features), dim=0
            )
            all_text_features = torch.cat(
                torch.distributed.nn.all_gather(text_features), dim=0
            )
        else:
            gathered_image_features = [
                torch.zeros_like(image_features) for _ in range(world_size)
            ]
            gathered_text_features = [
                torch.zeros_like(text_features) for _ in range(world_size)
            ]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
                self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = (
                    logit_scale * all_image_features @ all_text_features.T
                )
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2
        return total_loss

        
class SoftClipLoss(nn.Module):
    def __init__(self, temp=0.125, local_loss=False, gather_with_grad=False, cache_labels=False, rank=0, world_size=1, use_horovod=False):
        super().__init__()
        self.temp = temp
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, preds, targs, logit_scale):
        device = preds.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                preds,
                targs,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
                self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * preds @ all_text_features.T
                logits_per_text = logit_scale * targs @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * preds @ targs.T
            logits_per_text = logit_scale * targs @ preds.T

        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        target_similarity = (targs @ targs.T) / self.temp
        pred_target_similarity = (preds @ targs.T) / self.temp

        loss1 = -(pred_target_similarity.log_softmax(dim=-1) * target_similarity.softmax(dim=-1)).sum(dim=-1).mean()
        
        loss2 = -(pred_target_similarity.T.log_softmax(dim=-1) * target_similarity.softmax(dim=-1)).sum(dim=-1).mean()

        total_loss = (loss1 + loss2) / 2
        return total_loss
    
class OptimalTransportLoss(nn.Module):
    def __init__(self, epsilon=0.1, n_iters=100):
        super().__init__()
        self.epsilon = epsilon
        self.n_iters = n_iters

    def normalize_cost_matrix(self, cost_matrix):
        min_val = cost_matrix.min()
        max_val = cost_matrix.max()
        normalized_cost_matrix = (cost_matrix - min_val) / (max_val - min_val)
        return normalized_cost_matrix

    def compute_cost_matrix(self, features1, features2):
        b, c, n1, d = features1.shape
        _, _, n2, _ = features2.shape
        features1 = features1.to(torch.float16)
        features2 = features2.to(torch.float16)
        cost_matrix = torch.cdist(features1.reshape(-1, n1, d), features2.reshape(-1, n2, d), p=2).to(features1.device)
        return cost_matrix.view(b, c, n1, n2)

    @torch.no_grad()
    def sinkhorn_algorithm(self, cost_matrix, reg=0.1, n_iters=100, tol=1e-8):
        b, c, n1, n2 = cost_matrix.shape
        source = torch.ones(b, c, n1).to(cost_matrix.device)
        target = torch.ones(b, c, n2).to(cost_matrix.device)  

        transport_matrix = torch.exp(-cost_matrix / reg).to(cost_matrix.device)
        transport_matrix /= transport_matrix.sum(dim=(2, 3), keepdim=True)  # Normalize over n1 and n2 (for each batch and channel)

        # Reshape source and target
        source = source.view(b, c, -1, 1)
        target = target.view(b, c, 1, -1)
        for _ in range(n_iters):
            # Row normalization: update row ratios and transport matrix
            row_sum = transport_matrix.sum(dim=3, keepdim=True) + 1e-8  # Sum over n2 (columns)
            row_ratio = source / row_sum  # Normalize each row for each batch and channel
            transport_matrix *= row_ratio
            # Column normalization: update column ratios and transport matrix
            col_sum = transport_matrix.sum(dim=2, keepdim=True) + 1e-8 # Sum over n1 (rows)
            col_ratio = target / col_sum  # Normalize each column for each batch and channel
            transport_matrix *= col_ratio
            # Calculate the error: how well transport matrix sums to the source distribution
            err = torch.max(torch.abs(transport_matrix.sum(dim=2, keepdim=True) - source))
            if err < tol:
                break
        min_cost = torch.sum(transport_matrix * cost_matrix)
        return transport_matrix, min_cost
    # @torch.no_grad()
    # def sinkhorn_algorithm(self, cost_matrix):
    #     b, c, n1, n2 = cost_matrix.shape
    #     u = torch.ones(b, c, n1, device=cost_matrix.device) / n1
    #     v = torch.ones(b, c, n2, device=cost_matrix.device) / n2
    #     for _ in range(self.n_iters):
    #         u = 1 / (torch.sum(torch.exp(-cost_matrix / self.epsilon) * v.unsqueeze(2), dim=3) + 1e-8)
    #         v = 1 / (torch.sum(torch.exp(-cost_matrix / self.epsilon) * u.unsqueeze(3), dim=2) + 1e-8)
    #     T = torch.exp(-cost_matrix / self.epsilon) * u.unsqueeze(3) * v.unsqueeze(2)
    #     return T

    def max_norm_normalize(self, distances):
        max_val = distances.max()
        return distances / max_val

    def contrastive_loss(self, similarity, labels):
        return F.cross_entropy(similarity, labels)

    def forward(self, eeg_embeds, text_embeds):
        b = eeg_embeds.size(0)
        eeg_embeds_exp = eeg_embeds.unsqueeze(1).expand(b, b, -1, -1)  # [b, b, 25, 1024]
        text_embeds_exp = text_embeds.unsqueeze(0).expand(b, b, -1, -1)  # [b, b, 56, 1024]
        cost_matrix = self.compute_cost_matrix(eeg_embeds_exp, text_embeds_exp)  # [b, b, 25, 56]
        cost_matrix = self.normalize_cost_matrix(cost_matrix)
        T, min_cost = self.sinkhorn_algorithm(cost_matrix)  # T [b, b, 25, 56]
        distances = torch.sum(T * cost_matrix, dim=(2, 3))  # [b, b]
        distances_normalized = self.normalize_cost_matrix(distances)
        similarity = torch.exp(-distances_normalized)
        labels = torch.arange(b).to(similarity.device)  # [b]
        loss_i2t = self.contrastive_loss(similarity, labels)
        loss_t2i = self.contrastive_loss(similarity, labels)
        loss = (loss_i2t + loss_t2i) / 2
        return loss, similarity

    def ot_similarity(self, eeg_embeds, text_embeds):
        b1 = eeg_embeds.size(0)
        b2 = text_embeds.size(0)
        eeg_embeds_exp = eeg_embeds.unsqueeze(1).expand(b1, b2, -1, -1)  # [b, b, 25, 1024]
        text_embeds_exp = text_embeds.unsqueeze(0).expand(b1, b2, -1, -1)  # [b, b, 56, 1024]
        cost_matrix = self.compute_cost_matrix(eeg_embeds_exp, text_embeds_exp)  # [b, b, 25, 56]
        cost_matrix = self.normalize_cost_matrix(cost_matrix)
        T, min_cost = self.sinkhorn_algorithm(cost_matrix)  # T [b, b, 25, 56]
        distances = torch.sum(T * cost_matrix, dim=(2, 3))  # [b, b]
        distances_normalized = self.normalize_cost_matrix(distances)
        similarity = torch.exp(-distances_normalized)
        return similarity
# def compute_cost_matrix(features1, features2):
#     b, c, n1, d = features1.shape
#     _, _, n2, _ = features2.shape
#     # bmm
#     cost_matrix = torch.cdist(features1.reshape(-1, n1, d), features2.reshape(-1, n2, d), p=2)
#     return cost_matrix.view(b, c, n1, n2)

# def sinkhorn_algorithm(cost_matrix, epsilon=0.1, n_iters=100):
#     b, c, n1, n2 = cost_matrix.shape
#     u = torch.ones(b, c, n1, device=cost_matrix.device) / n1
#     v = torch.ones(b, c, n2, device=cost_matrix.device) / n2
#     for _ in range(n_iters):
#         u = 1 / (torch.sum(torch.exp(-cost_matrix / epsilon) * v.unsqueeze(2), dim=3) + 1e-8)
#         v = 1 / (torch.sum(torch.exp(-cost_matrix / epsilon) * u.unsqueeze(3), dim=2) + 1e-8)
#     T = torch.exp(-cost_matrix / epsilon) * u.unsqueeze(3) * v.unsqueeze(2)
#     return T

# def max_norm_normalize(distances):
#     max_val = distances.max()
#     return distances / max_val


# def ot_contrastive_loss(eeg_embeds, text_embeds, epsilon=0.1, n_iters=100):
#     b = eeg_embeds.size(0)
#     eeg_embeds_exp = eeg_embeds.unsqueeze(1).expand(b, b, -1, -1)  # [b, b, 25, 1024]
#     text_embeds_exp = text_embeds.unsqueeze(0).expand(b, b, -1, -1)  # [b, b, 56, 1024]
#     cost_matrix = compute_cost_matrix(eeg_embeds_exp, text_embeds_exp)  # [b, b, 25, 56]
#     T = sinkhorn_algorithm(cost_matrix, epsilon, n_iters) # T [b, b, 25, 56]
#     distances = torch.sum(T * cost_matrix, dim=(2, 3))  # [b, b]
#     distances_normalized = max_norm_normalize(distances)
#     similarity = torch.exp(-distances_normalized)
#     labels = torch.arange(b).to(similarity.device)  # [b]
#     loss_i2t = ce_loss(similarity, labels)
#     loss_t2i = ce_loss(similarity, labels)
#     loss = (loss_i2t + loss_t2i) / 2
#     return loss, similarity

# def ot_similarity(eeg_embeds, text_embeds, epsilon=0.1, n_iters=100):
#     b1 = eeg_embeds.size(0)
#     b2 = text_embeds.size(0)
#     eeg_embeds_exp = eeg_embeds.unsqueeze(1).expand(b1, b2, -1, -1)  # [b, b, 25, 1024]
#     text_embeds_exp = text_embeds.unsqueeze(0).expand(b1, b2, -1, -1)  # [b, b, 56, 1024]
#     cost_matrix = compute_cost_matrix(eeg_embeds_exp, text_embeds_exp)  # [b, b, 25, 56]
#     T = sinkhorn_algorithm(cost_matrix, epsilon, n_iters) # T [b, b, 25, 56]
#     distances = torch.sum(T * cost_matrix, dim=(2, 3))  # [b, b]
#     distances_normalized = max_norm_normalize(distances)
#     similarity = torch.exp(-distances_normalized)
#     return similarity
