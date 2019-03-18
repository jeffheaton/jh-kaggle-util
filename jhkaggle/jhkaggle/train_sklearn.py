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
from sklearn.ensemble import RandomForestRegressor

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

    def predict_model(self, clr, x):
        fit_type = jhkaggle.jhkaggle_config['FIT_TYPE']
        x = x.values.astype(np.float32)

        if fit_type == jhkaggle.const.FIT_TYPE_REGRESSION:
            return clr.predict(x)
        else:
            pred = clr.predict_proba(x)
            pred = np.array([v[1] for v in pred])
            return pred

# "all the time" to "always"
# reall short ones that are dead wrong

# [100]	train-logloss:0.288795	eval-logloss:0.329036
# [598]	train-logloss:0.152968	eval-logloss:0.296854
# [984]	train-logloss:0.096444	eval-logloss:0.293915

