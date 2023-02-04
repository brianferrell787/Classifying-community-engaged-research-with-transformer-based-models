# Classifying-community-engaged-research-with-transformer-based-models

We propose a novel approach to identifying and categorizing community-engaged studies by applying attention-based deep learning models to human subjects protocols that have been submitted to the university’s Institutional Review Board (IRB).

Attached and shown below is a timeline of this extensive project showing code and our academic accomplishments.

![alt text](https://community.vcu.edu/media/community2019/images/hero-images-/homepage-header.jpg)

https://community.vcu.edu/

# Timeline of papers
## Methods papers
### 1. Published before improvements (i.e. ClassificationProj.ipynb)
https://formative.jmir.org/2022/9/e32460

```latex
@article{ferrell2022attention,
  title={Attention-Based Models for Classifying Small Data Sets Using Community-Engaged Research Protocols: Classification System Development and Validation Pilot Study},
  author={Ferrell, Brian J and Raskin, Sarah E and Zimmerman, Emily B and Timberline, David H and McInnes, Bridget T and Krist, Alex H},
  journal={JMIR Formative Research},
  volume={6},
  number={9},
  pages={e32460},
  year={2022},
  publisher={JMIR Publications Inc., Toronto, Canada}
}
```

### 2. Published after initial improvements (fine-tuning strategies)

```latex
Waiting on acceptance
```

[Fine Tuning Strategies for Transformer-Based Models YouTube](https://www.youtube.com/watch?v=UcybL7v0OT8&t=356s)

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
https://www.cambridge.org/core/journals/journal-of-clinical-and-translational-science/article/developing-a-classification-system-and-algorithm-to-track-communityengaged-research-using-irb-protocols-at-a-large-research-university/7FCA3A9D8B6ACDAB664A4C45800100D2

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

Calibrated confidence scores were added, see example code below:

``` python
from netcal.metrics import ACE
import numpy as np
from netcal.binning import IsotonicRegression, BBQ, HistogramBinning, ENIR
from netcal.scaling import LogisticCalibration, TemperatureScaling, BetaCalibration
from netcal.presentation import ReliabilityDiagram

class Calibration():
    '''Set the initial binning and scaling methods'''
    def __init__(self):
        self.method1 = IsotonicRegression()
        self.method2 = BBQ()
        self.method3 = ENIR()
        self.method4 = HistogramBinning()
        self.method5 = LogisticCalibration()
        self.method6 = TemperatureScaling()
        self.method7 = BetaCalibration()
        
        
    def fit(self,probs, ground_truth):
        '''Set methods to global so it can be used in other functions for pritning and the fit the models'''
        global methods
        methods = [self.method1,self.method2,self.method3,self.method4,self.method5,self.method6,self.method7]
        for i in methods:
            i.fit(probs, ground_truth)
    
    def calibrate(self,bins=5):
        global calibrations
        '''Setting the bins to a random number, getting the ACE metric, uncalibrated score, and then calibrate the models'''
        self.bins = bins
        self.ace = ACE(self.bins)
        uncalibrated_score = self.ace.measure(probs,ground_truth)
        
        calibrated1 =  self.method1.transform(probs)
        calibrated2 =  self.method2.transform(probs)
        calibrated3 =  self.method3.transform(probs)
        calibrated4 =  self.method4.transform(probs)
        calibrated5 =  self.method5.transform(probs)
        calibrated6 =  self.method6.transform(probs)
        calibrated7 =  self.method7.transform(probs)
        
        calibrations = [calibrated1,calibrated2,calibrated3,calibrated4,calibrated5,calibrated6,calibrated7]
        
        #setting calscores brackets and calculating the ACE metric for models and print the scores plus the minimum score
        calscores = []
        for i in calibrations:
            calscores.append(self.ace.measure(i,ground_truth))
        for i,j in zip(methods,calscores):
            print(f"{type(i)}: {round(j,4)}\n") 
            
        print(f"Minimum ACE Metric: {round(np.min(calscores),4)}")
            
        print(f"Uncalibrated Score: {uncalibrated_score:.4f} \n")
        
        #this output is the type of model and the new probability values
        diagram = ReliabilityDiagram(self.bins,metric='ACE')
        for i,j in zip(methods,calibrations):
            print(f"{type(i)}: \n \n {j} \n {diagram.plot(j, ground_truth)}")
           
```

Example usage:

``` python
ground_truth = [2,2,2,2,0,0,1,2,2,0,1,1,1,1,1,2,2,2,2,1,2,2,2,2,2,2,2,1,1,0,0,0,0,2,0,0,0,0,0,1,1,1,1,0,2] #actual classes
ground_truth = np.array(ground_truth)

model = Calibration()
model.fit(probs,ground_truth) # probs are just the softmax outputs from the model predictions
model.calibrate(bins=3)
```
