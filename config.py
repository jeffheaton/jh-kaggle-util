# Jeff Heaton's Kaggle Utilities
# Copyright 2019 by Jeff Heaton, Open Source, Released under the Apache License
# For more information: https://github.com/jeffheaton/jh-kaggle-util
# 
# This file should be modified to define each new project.
import platform
from const import *

TRAIN_ID = "ID"
TEST_ID = "ID"

GOAL_MINIMIZE = False

ENCODING = 'utf-8'
TARGET_NAME = 'y'
FIT_TYPE = FIT_TYPE_REGRESSION
FINAL_EVAL = EVAL_R2


if "darwin" in platform.system().lower() :
    PATH = "/Users/jheaton/data/kaggle/mercedes"
elif "linux" in platform.system().lower() :
    PATH = "/home/jheaton/data/quora/mercedes"
else:
    PATH = "C:\\jth\\kaggle\\mercedes\\"
