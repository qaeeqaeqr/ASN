# AutoSewerNet

Official implementation of paper "Classification of sewer pipe defects based on an 1
automatically designed convolutional neural network"

### Environment

First, create a python environment:

``conda create -n asn python=3.8``

Then, install dependencies:

``pip install -r requirements.txt``


### Project

Our scripts are in folder [./AutoSewerNet](./AutoSewerNet)

Data in this paper is drawn by scipts in folder [./draw](./draw)

The number of samples od each class in the dataset is drawn by [draw_dataset](./draw/draw_dataset.py)

The comparison between AutoSewerNet and other algorithms is drawn by [result](./draw/draw_result.py)

The loss during training Super-net is drawn by [training_proc](./draw/draw_training_proc.py)

The scripts implement the following functions:

Constructing Super-net

Training Super-net [./AutoSewerNet/train_supernet.py](./AutoSewerNet/train_supernet.py)

Selecting subnets (results are in [sample_result.txt]('./sample_result.txt))

Training subnets and get AutoSewerNet[./AutoSewerNet/train_sampled.py](./AutoSewerNet/train_sampled.py)

Calculate F2CIW [./AutoSewerNet/eval_subnets_f2ciw.py](./AutoSewerNet/eval_subnets_f2ciw.py)