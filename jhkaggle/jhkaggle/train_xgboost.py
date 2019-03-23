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
import json


class TrainXGBoost(jhkaggle.util.TrainModel):
    def __init__(self, data_source, params, run_single_fold):
        super().__init__(data_source, run_single_fold)
        self.name="xgboost"
        self.params = params
        self.rounds = 10000
        self.early_stop = 50

    def train_model(self, x_train, y_train, x_val, y_val):
        print("Will train XGB for {} rounds, RandomSeed: {}".format(self.rounds, self.params['seed']))

        xg_train = xgb.DMatrix(x_train, label=y_train)

        if y_val is None:
            watchlist = [(xg_train, 'train')]
            model = xgb.train(self.params, xg_train, self.rounds, watchlist)
        else:
            early_stop = self.rounds if self.early_stop == 0 else self.early_stop
            xg_val = xgb.DMatrix(x_val, label=y_val)
            watchlist = [(xg_train, 'train'), (xg_val, 'eval')]
            model = xgb.train(self.params, xg_train, self.rounds, watchlist, early_stopping_rounds=early_stop)

        self.steps = model.best_iteration
        return model

    def predict_model(self, model, X_test):
        return model.predict(xgb.DMatrix(X_test))

    def feature_rank(self,output):
        rank = self.model.get_fscore()
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

    def save_model(self, path, name):
        print("Saving Model")

        self.model.save_model(os.path.join(path, name + ".bin"))

        meta = {
            'name': 'TrainXGBoost',
            'data_source': self.data_source,
            'params': self.params
        }
        
        with open(os.path.join(path,"meta.json"), 'w') as outfile:
            json.dump(meta, outfile)

    @classmethod
    def load_model(cls,path,name):
        root = jhkaggle.jhkaggle_config['PATH']
        model_path = os.path.join(root,path)
        meta_filename = os.path.join(model_path,"meta.json")
        with open(meta_filename, 'r') as fp:
            meta = json.load(fp)

        result = TrainXGBoost(meta['data_source'],None,False)
        result.model = xgb.Booster({'nthread':-1}) #init model
        result.model.load_model(os.path.join(model_path,name+".bin"))
        return result