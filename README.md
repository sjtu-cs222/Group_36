# Algorithm Project : Partial Restart of Neural Network

Group 36

***

Environment Requirements:

1. Pytorch  version = Lastest
2. Cudatoolkit=9.0

Our experiments can be divided into three parts.

1. Fine Tune
2. Select Initialization Methods
3. ReStart

First, you run in command line. 

```python
python alexnet_cifar10_own_train_baseline.py
python alexnet_cifar10_train_baseline.py
python mininet_cifar10_train_baseline.py
```

Then, you can get baseline models for these three networks. Move these model to the same folder in restart.

Run these in command line.

```python
python alexnet_cifar10_own_restart.py
python alexnet_cifar10_restart.py
python mininet_cifar10_restart.py
```

