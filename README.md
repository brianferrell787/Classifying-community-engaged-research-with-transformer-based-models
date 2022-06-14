# Classifying-community-engaged-research-with-transformer-based-models





# Timeline of papers
## Methods papers
### 1. Published before improvements (i.e. ClassificationProj.ipynb)

```latex
Waiting on acceptance
```

### 2. Published after initial improvements (fine-tuning strategies)

```latex
Waiting on acceptance
```

Layer freezing was added, see example code below:

```python
import logging
from statistics import mean

import pandas as pd
from sklearn.metrics import accuracy_score
import os
import wandb
from simpletransformers.classification import ClassificationArgs, ClassificationModel
path_to_file = r"C:/Users/ferre/OneDrive/Desktop/Bert"
model_args = ClassificationArgs()
model_args.evaluate_during_training = True
model_args.logging_steps = 10
model_args.num_train_epochs =  4
model_args.evaluate_during_training_steps =  500
model_args.save_eval_checkpoints =  False
model_args.save_steps =  500
model_args.manual_seed = 4
model_args.save_model_every_epoch = True
model_args.train_batch_size =  32
model_args.eval_batch_size =  8
model_args.output_dir = path_to_file
#model_args.best_model_dir =  path_to_file
model_args.overwrite_output_dir =  True
model_args.learning_rate =  3e-5
model_args.sliding_window = True
model_args.max_seq_length = 128
model_args.use_cuda = True
model_args.silent = True #Silent means no verbose
model_args.no_cache = True
model_args.no_save = False
model_args.reprocess_input_data =  True
model_args.fp16 = True
model_args.train_custom_parameters_only =  False
model_args.do_lower_case = True
#model_args.wandb_project = "NewdataBert"
#model_args.warmup_ratio = .1
#model_args.weight_decay = 0.95
model_args.custom_layer_parameters = [
    {
        "layer": 0,
        "lr": 0,
    },
    {
        "layer": 1,
        "lr": 0,
    },
     {
        "layer": 2,
        "lr": 0,
    },
     {
        "layer": 3,
        "lr": 0,
    },
     {
        "layer": 4,
        "lr":0,
    },
     {
        "layer":5,
        "lr": 0,
    },
     {
        "layer": 6,
        "lr":0,
    },
     {
        "layer": 7,
        "lr": 0,
    },
     {
        "layer": 8,
        "lr": 3e-5,
    },
     {
        "layer": 9,
        "lr": 3e-5,
    },
     {
        "layer": 10,
        "lr": 3e-5,
    },
     {
        "layer": 11,
        "lr": 3e-5,
    },
     {
        "layer":12,
        "lr":  3e-5,
    },
]


# Create a ClassificationModel
model = ClassificationModel(
    "bert", 
    "bert-base-uncased",
    num_labels=3, #this would be 6 if we were using the 6 class spectrum 
    weight=[ 1.10207336523126,   1.0501519756838906,  0.8769035532994924],
    args=model_args
)
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Train model
model.train_model(training,eval_df=test)
```
## Evaluation papers
### 1. Published after initial improvements (fine-tuning strategies)

```latex
@article{zimmerman2022developing,
  title={Developing a classification system and algorithm to track community-engaged research using IRB protocols at a large research university},
  author={Zimmerman, Emily B and Raskin, Sarah E and Ferrell, Brian and Krist, Alex H},
  journal={Journal of Clinical and Translational Science},
  volume={6},
  number={1},
  year={2022},
  publisher={Cambridge University Press}
}
```

### 4. Published after additional improvements (new data, calibrated confidence scores)

```latex
Waiting on acceptance
```

