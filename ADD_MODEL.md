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
