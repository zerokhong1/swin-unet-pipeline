import random
from train import get_args
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torch.optim import Adam,SGD
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
import pandas as pd
# import the necessary packages
import os
# import the necessary packages
from torch.utils.data import Dataset
import cv2
import gc
# base path of the dataset
import shutil
import sys
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

"""Phần I: Xét các tham số"""
# THAM SỐ HẰNG SỐ
# Đặt seed để đảm bảo tái hiện kết quả
SEED=42
torch.manual_seed(SEED)
# THAM SỐ VỪA LÀ HẰNG SỐ VỪA THAY ĐỔI
INIT_LR = 0.0001
# lr0= INIT_LR
BATCH_SIZE = 8
WEIGHT_DECAY=1e-6
# weight_decay = 1e-6  # Regularization term to prevent overfitting
INPUT_IMAGE_WIDTH = 256
INPUT_IMAGE_HEIGHT = 256
NUM_CLASSES = 1


"""Phần II: Xử lý logic"""
args = get_args()

#  Tham số trường hợp:
augment = args.augment
# tham số vừa là hằng số vừa thay đổi:
lr0 = args.lr0 if args.lr0 else INIT_LR
bach_size = args.batchsize if args.batchsize else BATCH_SIZE
weight_decay = args.weight_decay if args.weight_decay else WEIGHT_DECAY
input_image_width, input_image_height = args.img_size if args.img_size else [INPUT_IMAGE_WIDTH,INPUT_IMAGE_HEIGHT]
numclass = args.numclass if args.numclass else NUM_CLASSES
# THAM SỐ LUÔN THAY ĐỔI THEO nhap.py
NUM_EPOCHS = args.epoch
T_max = NUM_EPOCHS  # T_max là số epoch bạn muốn dùng cho giảm lr
lr_min = 0.0001  # lr_min là learning rate tối thiểu
# CÁC THAM SỐ KHÁC
DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = True if str(DEVICE) == "cuda:0" else False
DATASET_PATH = args.data
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VALID_PATH = os.path.join(DATASET_PATH, "valid")
TEST_PATH = os.path.join(DATASET_PATH, "test")

# define the path to the images and masks dataset
IMAGE_TRAIN_PATH = os.path.join(TRAIN_PATH, "images")
MASK_TRAIN_PATH = os.path.join(TRAIN_PATH, "masks")

IMAGE_VALID_PATH = os.path.join(VALID_PATH, "images")
MASK_VALID_PATH = os.path.join(VALID_PATH, "masks")

IMAGE_TEST_PATH = os.path.join(TEST_PATH, "images")
MASK_TEST_PATH = os.path.join(TEST_PATH, "masks")

BASE_OUTPUT = args.saveas if  args.saveas else "output"
# Tham số của model
# model = 
# print("lr0",lr0)
# print("bach_size",bach_size)
# print("weight_decay",weight_decay)
# print("input_image_width",input_image_width)
# print("input_image_height",input_image_height)
# print("numclass",numclass)
# print("NUM_EPOCHS",NUM_EPOCHS)






# DATASET_PATH = sys.argv[1] if len(sys.argv) > 1 else "format_data"

# TRAIN_PATH = os.path.join(DATASET_PATH, "train")
# VALID_PATH = os.path.join(DATASET_PATH, "valid")

# # define the path to the images and masks dataset
# IMAGE_TRAIN_PATH = os.path.join(TRAIN_PATH, "images")
# MASK_TRAIN_PATH = os.path.join(TRAIN_PATH, "masks")

# IMAGE_VALID_PATH = os.path.join(VALID_PATH, "images")
# MASK_VALID_PATH = os.path.join(VALID_PATH, "masks")



# # define the test split
# TEST_SPLIT = 0.25
# # determine the device to be used for training and evaluation
# # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # determine if we will be pinning memory during data loading
# # PIN_MEMORY = True if DEVICE == "cuda" else False
# PIN_MEMORY = True if str(DEVICE) == "cuda:0" else False
# print(DEVICE)
# print(PIN_MEMORY)
# # define the number of channels in the input, number of classes,
# # and number of levels in the U-Net model
# NUM_CHANNELS = 3

# NUM_LEVELS = 3
# # initialize learning rate, number of epochs to train for, and the
# # batch size
# # INIT_LR = 0.0001
# NUM_EPOCHS = 2

# # define the input image dimensions
# INPUT_IMAGE_WIDTH = 256
# INPUT_IMAGE_HEIGHT = 256
# # define threshold to filter weak predictions
# THRESHOLD = 0.5
# # define the path to the base output directory
# BASE_OUTPUT = "output"

# # define the path to the output serialized model, model training
# # plot, and testing image paths
# MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_tgs_salt.pth")
# PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
