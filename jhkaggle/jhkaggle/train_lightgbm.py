# Jeff Heaton's Kaggle Utilities
# Copyright 2019 by Jeff Heaton, Open Source, Released under the Apache License
# For more information: https://github.com/jeffheaton/jh-kaggle-util
# 
# Train for Light GBM.
# https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
import pandas as pd
import numpy as np
import time
import os
import zipfile
import operator
from sklearn.metrics import log_loss
import lightgbm as lgb
import scipy
import jhkaggle.util

class TrainLightGBM(jhkaggle.util.TrainModel):
    def __init__(self, data_source, params, run_single_fold):
        super().__init__(data_source, run_single_fold)
        self.name="lgb"
        self.params = params
        self.rounds = 25000
        self.early_stop = 50

    def train_model(self, x_train, y_train, x_val, y_val):
        print("Will train LightGB for {} rounds".format(self.rounds))
        #x_train = scipy.stats.zscore(x_train)

        lgb_train = lgb.Dataset(x_train, label=y_train)
         
        if y_val is None:
            clr = lgb.train(self.params, lgb_train, self.rounds, valid_sets = [lgb_train], verbose_eval=100, early_stopping_rounds = 2000)
        else:
            early_stop = self.rounds if self.early_stop == 0 else self.early_stop
            lgb_val = lgb.Dataset(x_val, label=y_val)
            clr = lgb.train(self.params, lgb_train, self.rounds, valid_sets = [lgb_train, lgb_val], verbose_eval=100, early_stopping_rounds = 2000)
  
        self.steps = clr.best_iteration
        print(f"Best: {self.steps}")
        return clr

    def predict_model(self, clr, X_test):
        return clr.predict(X_test)

    def feature_rank(self,output):
        return "Unknown"


# bst = lgb.train(param, train_data, num_round, valid_sets=valid_sets, early_stopping_rounds=10)
# bst.save_model('model.txt', num_iteration=bst.best_iteration)

