# Jeff Heaton's Kaggle Utilities
# Copyright 2019 by Jeff Heaton, Open Source, Released under the Apache License
# For more information: https://github.com/jeffheaton/jh-kaggle-util
# 
# Train for XGBoost.
import pandas as pd
import numpy as np
import xgboost as xgb
import time
import os
import zipfile
import operator
from sklearn.metrics import log_loss
import scipy
import jhkaggle.util

class TrainXGBoost(jhkaggle.util.TrainModel):
    def __init__(self, data_source, params, run_single_fold):
        super().__init__(data_source, run_single_fold)
        self.name="xgboost"
        self.params = params
        self.rounds = 10000
        self.early_stop = 50

    def train_model(self, x_train, y_train, x_val, y_val):
        print("Will train XGB for {} rounds, RandomSeed: {}".format(self.rounds, self.params['seed']))
        #x_train = scipy.stats.zscore(x_train)

        xg_train = xgb.DMatrix(x_train, label=y_train)

        if y_val is None:
            watchlist = [(xg_train, 'train')]
            clr = xgb.train(self.params, xg_train, self.rounds, watchlist)
        else:
            early_stop = self.rounds if self.early_stop == 0 else self.early_stop
            xg_val = xgb.DMatrix(x_val, label=y_val)
            watchlist = [(xg_train, 'train'), (xg_val, 'eval')]
            clr = xgb.train(self.params, xg_train, self.rounds, watchlist, early_stopping_rounds=early_stop)

        self.steps = clr.best_iteration
        return clr

    def predict_model(self, clr, X_test):
        return clr.predict(xgb.DMatrix(X_test))

    def feature_rank(self,output):
        rank = self.clr.get_fscore()
        rank_sort = sorted(rank.items(), key=operator.itemgetter(1))
        rank_sort.reverse()
        for f in rank_sort:
            output += str(f) + "\n"

        return output

    def run_cv(self):
        self._run_startup()
        print("Running CV to determine optional number of rounds")
        xg_train = xgb.DMatrix(self.x_train, label=self.y_train)
        cvresult = xgb.cv(self.params, xg_train, num_boost_round=self.rounds,nfold=self.num_folds,
                verbose_eval = True, early_stopping_rounds=self.early_stop)
        self.rounds=cvresult.shape[0]
        #self.score = cvresult.iloc[-1]['test-logloss-mean']
        self.score = cvresult.iloc[-1]['test-rmse-mean']
        self.scores = [self.score]
        print("Should use {} rounds.".format(self.rounds))
        return self.score, self.rounds