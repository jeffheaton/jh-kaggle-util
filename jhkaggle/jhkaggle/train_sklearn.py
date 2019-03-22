# Jeff Heaton's Kaggle Utilities
# Copyright 2019 by Jeff Heaton, Open Source, Released under the Apache License
# For more information: https://github.com/jeffheaton/jh-kaggle-util
# 
# Train from a SK-Learn model.

import jhkaggle.util
import tensorflow as tf
import tensorflow.contrib.learn as learn
import scipy.stats
import numpy as np
import time
import sklearn
import os
import json
from sklearn.ensemble import RandomForestRegressor
import pickle

class TrainSKLearn(jhkaggle.util.TrainModel):
    def __init__(self, data_set, name, alg, run_single_fold):
        super().__init__(data_set, run_single_fold)
        self.name=name
        self.alg=alg
        self.early_stop = 50
        self.params = str(alg)

    def train_model(self, x_train, y_train, x_val, y_val):
        print("Training SKLearn model: {}".format(self.alg))

        x_train = x_train.values.astype(np.float32)
        y_train = y_train.values.astype(np.int32)

        #x_val = x_val.as_matrix().astype(np.float32)
        #y_val = y_val.as_matrix().astype(np.int32)

        self.alg.fit(x_train, y_train)

        self.steps = 0

        #self.classifier = clr.best_iteration
        return self.alg

    def predict_model(self, model, x):
        fit_type = jhkaggle.jhkaggle_config['FIT_TYPE']

        if fit_type == jhkaggle.const.FIT_TYPE_REGRESSION:
            return model.predict(x)
        else:
            pred = model.predict_proba(x)
            pred = np.array([v[1] for v in pred])
            return pred

    @classmethod
    def load_model(cls,path,name):
        root = jhkaggle.jhkaggle_config['PATH']
        model_path = os.path.join(root,path)
        meta_filename = os.path.join(model_path,"meta.json")
        with open(meta_filename, 'r') as fp:
            meta = json.load(fp)

        result = cls(meta['data_source'],meta['params'],None,False)
        with open(os.path.join(model_path, name + ".pkl"), 'rb') as fp:  
            result.model = pickle.load(fp)
        return result
