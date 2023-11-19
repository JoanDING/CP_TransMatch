# This is the repo for the paper: [Modeling Multi-Relational Connectivity for Personalized Fashion Matching](https://dl.acm.org/doi/pdf/10.1145/3581783.3612583)

## Requirements
1. OS: Ubuntu 22.04
2. python 3.8.16
3. Supported (tested) CUDA Versions: V11.1
4. python modules: refer to the modules in [requirements.txt](https://github.com/JoanDING/CP_TransMatch/blob/main/requirements.txt)


## Code Structure
1. The entry script for training and evaluation is: [main.py](https://github.com/JoanDING/CP_TransMatch/blob/main/main.py)
2. The config file is: [train_model_config.yaml](https://github.com/JoanDING/CP_TransMatch/blob/main/config.yaml)
4. The script for data preprocess and dataloader: [utility.py](https://github.com/JoanDING/CP_TransMatch/blob/main/utility.py), [dataloader.py](https://github.com/JoanDING/CP_TransMatch/blob/main/dataloader.py)
5. The CP_TransMatch model: CP_TransMatch.py](https://github.com/JoanDING/CP_TransMatch/blob/main/CP_TransMatch.py)
6. Utility codes for evaluation: [eval_util.py](https://github.com/JoanDING/CP_TransMatch/blob/main/eval_utils.py)
7. Utility codes for knowledge graph-relevant functions: [eval_util.py](https://github.com/JoanDING/CP_TransMatch/blob/main/kg_utils.py)
8. The model folder: ./model/
9. The recommendation results in the evaluation are recorded in ./results/
10. The evaluation performance during training is saved in ./performance/
11. The ./model, ./performance, ./results files will be generated automatically when first time runing the codes. 


## How to Run
1. We prepare the processed iqon_s dataset as an example to illustrate the CP_TransMatch method, which you can find [here](https://drive.google.com/file/d/1h9GnR354HRUuhufNoJB0YtjedQeduYJE/view?usp=drive_link). Download and unzip the data file, put it into the roo_datapath which you have assign in data_config.yaml. iqon_s is processed based on the dataset provided by <https://github.com/SparkJiao/MG-PFCM_outfit_rec> related to the [paper](https://dl.acm.org/doi/pdf/10.1145/3477495.3532038). Codes for the pre-processing can be found in utility.py, you can also download the original data and try to process it by yourself.
2. Settings in the configure file train_model_config.yaml are all experimental settings mainly for the training process and the model. Before getting started, you need to revise the root_path for saving your experimental rsults. Settings related to dataset is in data_config.yaml, in which you also need to revise the root_datapath into the path of your data. We offer one hyper-parameters feasibly to change with the command line, which is the dataset (-d) for the experiment. You can also specify which gpu device (-g) to use in the experiments. 

3. Run the training and evaluation with the specified hyper-parameters by the command: 
    ```
    python main.py -d=iqon_s -g=0
    ```

4. The performance of the model is saved in ./performance. You can get into the folder and check the detailed training process of any finished experiments (Compared with the tensorboard log save in ./logs, it is just the txt-version human-readable training log). To quickly check the results for all implemented experiments, you can also print the results of all experiments for specific dataset (say "iqon") in a table format on the terminal screen by running: 
    ```
    python get_performance.py -d=iqon_s
    ```
