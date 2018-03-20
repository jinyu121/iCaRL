# iCaRL: incremental Class and Representation Learning
-- a modified version

For pure features, we cut off the ResNet and the net-training process.

## Requirements

- Numpy
- Scipy
- pyyaml
- easydict
- h5py (read `.mat` file with new format)
- sklearn (shuffle data)
- tqdm (show a process bar)

## How to use

### Extract features from your dataset

### Generate data file

1. Modify `conf/demo.yml`
1. And perhaps you have to modify `data_gen.py` to adjust to the data structure in your `.mat` file 
1. Run `python3 data_gen.py`

It will generate a data file in `data` folder with the format of 

```
[
    {   // Group #1
        "train_feature": train_feature, 
        "train_label": train_label,
        "eval_feature": eval_feature, 
        "eval_label": eval_label
    },
    {   // Group #2
        "train_feature": train_feature, 
        "train_label": train_label,
        "eval_feature": eval_feature, 
        "eval_label": eval_label
    }
    ...And more groups...
]
``` 

### Train and test

1. Modify `conf/demo.yml`
1. Run `python3 main.py`

Because of the cutting off of network, we just show the accuracy of iCaRL and NCM.