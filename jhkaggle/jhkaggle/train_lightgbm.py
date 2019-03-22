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
import json

class TrainLightGBM(jhkaggle.util.TrainModel):
    def __init__(self, data_source, params, run_single_fold):
        super().__init__(data_source, run_single_fold)
        self.name = "lgb"
        self.params = params
        self.rounds = 25000
        self.early_stop = 50

    def train_model(self, x_train, y_train, x_val, y_val):
        print("Will train LightGB for {} rounds".format(self.rounds))

        lgb_train = lgb.Dataset(x_train, label=y_train)
         
        if y_val is None:
            model = lgb.train(self.params, lgb_train, self.rounds, valid_sets = [lgb_train], verbose_eval=100, early_stopping_rounds = 2000)
        else:
            early_stop = self.rounds if self.early_stop == 0 else self.early_stop
            lgb_val = lgb.Dataset(x_val, label=y_val)
            model = lgb.train(self.params, lgb_train, self.rounds, valid_sets = [lgb_train, lgb_val], verbose_eval=100, early_stopping_rounds = 2000)
  
        self.steps = model.best_iteration
        print(f"Best: {self.steps}")
        return model

    def predict_model(self, model, X_test):
        return model.predict(X_test)

    def feature_rank(self,output):
        importance = self.model.feature_importance()
        top_importance = max(importance)
        importance = [x/top_importance for x in importance]
        importance = sorted(zip(self.x_train.columns, importance), key=lambda x: x[1])
        importance = sorted(importance, key=lambda tup: -tup[1])
        
        for row in importance:
            output += f"{row}\n"
        return output

    def save_model(self, path, name):
        print("Saving Model")
        self.model.save_model(os.path.join(path,name + ".txt"))
        meta = {
            'name': 'TrainLightGBM',
            'data_source': self.data_source,
            'params': self.params
        }
        
        with open(os.path.join(path,"meta.json"), 'w') as outfile:
            json.dump(meta, outfile)

    def load_model(path,name):
        root = jhkaggle.jhkaggle_config['PATH']
        model_path = os.path.join(root,path)
        meta_filename = os.path.join(model_path,"meta.json")
        with open(meta_filename, 'r') as fp:
            meta = json.load(fp)
        result = TrainLightGBM(meta['data_source'],meta['params'],False)
        result.model = lgb.Booster(model_file=os.path.join(model_path,name+".txt"))
        return result
        