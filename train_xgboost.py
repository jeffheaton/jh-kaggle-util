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

from util import *

class TrainXGBoost(TrainModel):
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

COMMON = {
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

# params = {'scale_pos_weight': 1, 'max_depth': 2, 'subsample': 0.9, 'seed': 42, 'gamma': 0.0, 'colsample_bytree': 0.9, 'learning_rate': 0.005, 'reg_alpha': 1, 'silent': 1, 'min_child_weight': 9}
#params = {'colsample_bytree': 0.9, 'gamma': 0.0, 'learning_rate': 0.01, 'min_child_weight': 9, 'reg_alpha': 0.01, 'seed': 42, 'subsample': 0.9, 'scale_pos_weight': 1, 'max_depth': 2}
#params = {'colsample_bytree': 0.6, 'min_child_weight': 9, 'subsample': 0.6, 'max_depth': 2, 'reg_alpha': 0.1, 'seed': 42, 'learning_rate': 0.005, 'gamma': 0.0}
#params = {'learning_rate' : 0.01, 'colsample_bytree': 0.2, 'subsample' : 1.0, 'max_depth' : 7, 'min_child_weight': 10, 'seed' : 4242}

#params = {'min_child_weight': 7, 'reg_alpha': 0.01, 'gamma': 0.0, 'max_depth': 3, 'subsample': 0.7, 'scale_pos_weight': 1, 'learning_rate': 0.01, 'seed': 42, 'colsample_bytree': 0.6}
#params = {'base_score': 100.669318128,'reg_alpha': 0.1, 'colsample_bytree': 0.6, 'learning_rate': 0.005, 'gamma': 0.0, 'seed': 42, 'min_child_weight': 9, 'max_depth': 2, 'subsample': 0.6, 'eval_metric': 'rmse'}
params = {'base_score': 100.669318128, 'learning_rate': 0.005, 'scale_pos_weight': 1, 'colsample_bytree': 0.7, 'min_child_weight': 9, 'subsample': 0.6, 'max_depth': 2, 'silent': 1, 'gamma': 0.0, 'seed': 42, 'reg_alpha': 0.01}
params = {'scale_pos_weight': 1, 'seed': 42, 'learning_rate': 0.005, 'base_score': 100.669318128, 'colsample_bytree': 0.6, 'max_depth': 2, 'gamma': 0.0, 'reg_alpha': 1, 'silent': 1, 'subsample': 0.7, 'min_child_weight': 9}
params = {'learning_rate':0.0045,'base_score': 100.669318128,'seed':4242}


params = {'max_depth': 2, 'subsample': 0.9, 'reg_alpha': 100, 'gamma': 0.0, 'min_child_weight': 7, 'seed': 4242, 'colsample_bytree': 0.9, 'silent': 1, 'base_score': 100.669318128, 'learning_rate': 0.0045}


params = {**params, **COMMON}
print(params)

train = TrainXGBoost("1",params=params,run_single_fold=False)
train.early_stop = 50
train.rounds = 10000
#train.run_cv()
train.run()
