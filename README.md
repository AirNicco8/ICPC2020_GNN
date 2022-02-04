# NLP_project_pyGNN
This repo contains the code for Unibo course NLP project 21/22, our aim is Code Summarization on a Python [dataset](https://github.com/github/CodeSearchNet) using the architecture proposed by the ICPC 2020 paper "Improved Code Summarization via a Graph Neural Network" - [arxiv](https://arxiv.org/abs/2004.02843)

The reproducibility package has three parts:
1. the code found in this repository
2. the unprocessed data, trained models, predictions, and tokenizer (.tok) files can be downloaded [HERE](https://drive.google.com/drive/folders/1SZZjK_YJGcH9snEr2T6hzejDEA07oPAo?usp=sharing)
3. the fully processed data (as a pkl file) can be downloaded [HERE](https://drive.google.com/drive/folders/1SZZjK_YJGcH9snEr2T6hzejDEA07oPAo?usp=sharing)

This code uses Keras v2.3.1 and Tensorflow v1.15.2 

## Processing Files
Our processing files assume to have 3 dataframes saved in .pkl format, obtained by the starting challenge mentioned before. {train_py, val_py, test_py}.pkl

After that the python scripts have to be launched following the numbers order, with the exception that 2.5, 3 and 3.5 steps can be skipped by downloading the tokenizers (.tok files) from the shared drive folder.

The final output produced by the processing files is 'dataset.pkl', which will be fed into the model in train.py

## Running the code and models

To run the trained models from the paper download the three parts of the reproducibility package and run predict.py. Predict.py takes the path to the model file as a positional argument and will output the prediction file to ./modelout/predictions.

`python3 predict.py {path to model} --gpu 0 --modeltype {model type: codegnngru|codegnnbilstm|codegnndense} --data {path to data download}`

`python3 predict.py ./mymodels/codegnngru.h5 --gpu 0 --modeltype codegnngru --data ./mydata`

To train a new model run train.py with the modeltype and gpu options set.

`python3 train.py --gpu 0 --modeltype codegnnbilstm --data ./mydata`

## Cite the forked repo
```
@inproceedings{
leclair2020codegnn,
title={Improved Code Summarization via a Graph Neural Network},
author={Alex LeClair, Sakib Haque, Lingfei Wu, Collin McMillan},
booktitle={2020 IEEE/ACM International Conference on Program Comprehension},
year={2020},
month={Oct.},
doi={10.1145/3387904.3389268}
ISSN={978-1-4503-7958-8/20/05}
}
```
