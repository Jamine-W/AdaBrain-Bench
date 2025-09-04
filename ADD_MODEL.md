# Add a New Model
New model can be added by following the steps.

---

### 2. Re-define model to fit our framework
First, place your model files in the `models/` folder. Next, make the following modifications in the python file `run_finetuning.py` (Take CBraMod as an example).

**Modification 1**: Import the model
   ```python
   from models.cbramod import CBraMod
   ```

**Modification 2**: Create a model wrapper class **Ada_ModelX** that includes model-specific preprocessing and feedforward steps. The following is an example of **Ada_CBraMod**:
   
   <!-- **Step 1:** Create a wrapper class  -->

   <!-- Initialize the original model

   ```python
   model = CBraMod()
   ``` -->

   <!-- **Step 2:** *(Optional)* load pre-trained weights  
   ```python
   if from_pretrain:
       print("Load ckpt from %s" % fintune_list[args.model_name])
       model.load_state_dict(
           torch.load(fintune_list[args.model_name], map_location=torch.device('cpu'))
       )
   ```
   > `finetune_list` contains the mapping defined earlier; `from_pretrain` is passed later. -->

   <!-- **Step 3:** Replace the original task head with `nn.Identity()` and register a new task head  
   ```python
   model.proj_out = nn.Identity()
   self.task_head = nn.Identity()
   ```
   > The specific task head will be added later. -->

   <!-- **Step 4:** Store the model and any additional modules; your original model is defined as `self.main_model`.  
   ```python
   self.main_model = model
   ```
   > If your model needs extra layers (e.g., EEGPT requires a 1-D convolution before the main model), add them here:  
   ```python
   self.chan_conv = Conv1dWithConstraint(len(ch_names), chans_num, 1, max_norm=1)
   ``` -->

   <!-- **Step 5:** Define the `forward()` method.  
   > Implement the full pipeline from input to final output. We will directly use `output = model(input)` to obtain your model's output in subsequent steps. -->

   <!-- For **CBraMod** the data must be reshaped to `[batch_size, channel, time, 200]`:  
   ```python
   def forward(self, x):
       b, n, t = x.shape
       x = x.reshape(b, n, -1, 200)
       y = self.main_model(x)
       return self.task_head(y) -->
   <!-- ``` -->
   <!-- > `y` is the raw output of your model, and the final result returned will be the value after passing through `self.task_head`. -->

```python
class Ada_CBraMod(nn.Module):
    def __init__(self, args, from_pretrain=False):
        super().__init__()
        # step 1: initialize the model
        model = CBraMod()
        # step 2: load the pretrained weight (optional)
        if from_pretrain:
            print("Load ckpt from %s" % fintune_list[args.model_name])
            model.load_state_dict(
                torch.load(fintune_list[args.model_name], map_location=torch.device('cpu'))
            )
        # step 3: remove the original task head  
        model.proj_out = nn.Identity()
        # step 4: register a new one, which will be specified later
        self.task_head = nn.Identity()
        # step 5: wrap the model
        self.main_model = model

    def forward(self, x):
        # step 6: adding model-specific preprocessing steps
        b, n, t = x.shape
        x = x.reshape(b, n, -1, 200)
        # step 7: model input and output
        output = self.main_model(x)
        # step 8: raw output->task output
        output = self.task_head(output)
        return output
```

---

### 3. Instantiate Your Model for Different Tasks  
In the `get_models()` function (`run_finetuning.py`), instantiate the model and attach the appropriate task head.

```python
model = Ada_CBraMod(args)
if args.task_mod == 'Classification':
    model.task_head = LinearWithConstraint(
        len(ch_names) * num_t * 200, args.nb_classes, max_norm=1, flatten=1
    )
elif args.task_mod == 'Regression':
    model.task_head = RegressionLayers(
        input_dim=(len(ch_names) * num_t) * 200,
        hidden_dim=200,
        output_dim=1,
        flatten=1
    )
elif args.task_mod == 'Retrieval':
    model.task_head = LinearWithConstraint(
        len(ch_names) * num_t * 200, 1024, max_norm=1, flatten=1
    )
```

We provide two ready-made heads whose implementations can be found in `run_finetuning.py`.  
Pick one according to the downstream task:

| **Task Head**           | **Recommended For** |
|:-----------------------:|:-------------------:|
| `LinearWithConstraint`  | **Classification** and **Retrieval** |
| `RegressionLayers`         | **Regression** |

Both heads expose arguments that let you decide whether to **flatten**, **average**, **remove the cls_token**, or apply other pre-processing before the final projection.
