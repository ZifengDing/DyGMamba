# DyGMamba: Efficiently Modeling Long-Term Temporal Dependency on Continuous-Time Dynamic Graphs with State Space Models
This repository is built for the paper DyGMamba: Efficiently Modeling Long-Term Temporal Dependency on Continuous-Time Dynamic Graphs with State Space Models


## Dynamic Graph Learning Models
The directory contains seven benchmark dataset and our work *DyGMamba*, which is an efficient continuous-time dynamic graph(CTDG) representation learning model that can capture long-term temporal dependencies. It also contains previous CTDG learning methods, including *JODIE*,*DyRep*,*TGAT*,*TGN*,*CAWN*,*TCL*,*GrapnMixer* and *DyGFormer*. 


## Evaluation Tasks

The code supports dynamic link prediction under both transductive and inductive settings with three (i.e., random, historical, and inductive) negative sampling strategies.



## Environments

[PyTorch](https://pytorch.org/),
[numpy](https://github.com/numpy/numpy),
[pandas](https://github.com/pandas-dev/pandas),
[tqdm](https://github.com/tqdm/tqdm), 
[tabulate](https://github.com/astanin/python-tabulate), and
[mamba-ssm](https://github.com/state-spaces/mamba)

To install the packages, run

```{bash}
pip install -r requirements.txt
```

## Executing Scripts
We test our model using seven datasets: *wikipedia*, *reddit*, *mooc*, *lastfm*, *enron*, *SocialEvo* and *uci*. They can be downloaded. To run the training or evaluation, please [download](https://zenodo.org/record/7213796#.Y1cO6y8r30o) the datasets and put the unzipped file in ```processed_data``` folder. 

For example, the directory of *uci* dataset should be structured as follows:

```plaintext
project-root/
└── processed_data/
    └── uci/
        ├── ml_uci.csv
        ├── ml_uci.npy
        └── ml_uci_node.npy
```

### Scripts for Dynamic Link Prediction
Dynamic link prediction could be performed on all the seven datasets. 
If you want to load the best model configurations determined by the grid search, please set the *load_best_configs* argument to True.
#### Model Training
* Example of training *DyGMamba* on *Wikipedia* dataset:
```{bash}
python train_link_prediction.py --dataset_name wikipedia --model_name DyGMamba --patch_size 2 --max_input_sequence_length 64 --num_runs 5 --gpu 0
```
* If you want to use the best model configurations to train *DyGMamba* on *Wikipedia* dataset, run
```{bash}
python train_link_prediction.py --dataset_name wikipedia --model_name DyGMamba --load_best_configs --num_runs 5 --gpu 0
```
#### Model Evaluation
Three (i.e., random, historical, and inductive) negative sampling strategies can be used for model evaluation.
* Example of evaluating *DyGMamba* with *random* negative sampling strategy on *Wikipedia* dataset:
```{bash}
python evaluate_link_prediction.py --dataset_name wikipedia --model_name DyGMamba --patch_size 2 --max_input_sequence_length 64 --negative_sample_strategy random --num_runs 5 --gpu 0
```
* If you want to use the best model configurations to evaluate *DyGMamba* with *random* negative sampling strategy on *Wikipedia* dataset, run
```{bash}
python evaluate_link_prediction.py --dataset_name wikipedia --model_name DyGMamba --negative_sample_strategy random --load_best_configs --num_runs 5 --gpu 0
```
