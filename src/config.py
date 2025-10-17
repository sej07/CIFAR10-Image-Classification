import torch
import os

#device
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

#paths
DATA_DIR = './data'
MODEL_DIR = './models'
RESULTS_DIR = './results'

#data parameters
BATCH_SIZE = 128
NUM_WORKERS = 2
TRAIN_SPLIT = 0.9

#model architecture
INPUT_CHANGES = 3
NUM_CLASSES = 10
DROPOUT_RATE = 0.5

#training hyperparameters
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-4

#normalization 
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

#reproducibility
RANDOM_SEED = 42