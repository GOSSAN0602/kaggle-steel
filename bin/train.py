import os
import cv2
import pdb
import time
import warnings
import random
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
from matplotlib import pyplot as plt
from albumentations import HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise
from albumentations.pytorch import ToTensor
import sys
sys.path.append('./')
from libs.dataloader import *
from libs.mask_utils import *
from libs.trainer import *
from libs.utils import *
from libs.models import Unet
from libs.plot import plot
warnings.filterwarnings("ignore")
seed = 69
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# folder paths
sample_submission_path = '../input/sample_submission.csv'
test_data_folder = "../input/test_images"

# define model
model = Unet("resnet101", encoder_weights="imagenet", classes=4, activation=None)

# define model_trainer
model_trainer = Trainer(model)

# train start
model_trainer.start()

# plot training process
losses = model_trainer.losses
dice_scores = model_trainer.dice_scores # overall dice
iou_scores = model_trainer.iou_scores

plot(losses, "BCE loss")
plot(dice_scores, "Dice score")
plot(iou_scores, "IoU score")
