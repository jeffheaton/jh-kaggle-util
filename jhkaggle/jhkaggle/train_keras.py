import jhkaggle.util
import tensorflow as tf
import scipy.stats
import numpy as np
import time
import os
import json
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model


class TrainKeras(jhkaggle.util.TrainModel):
    def __init__(self, data_source, run_single_fold):
        super().__init__(data_source, run_single_fold)
        self.name="keras"
        self.params = []
        self.early_stop = 50

    def define_neural_network(self, x):
        # Modify this to define the type of neural network, hidden layers, etc.
        model = Sequential()
        model.add(Dense(20, input_dim=x.shape[1], activation='relu'))
        model.add(Dense(10, activation='relu'))
        return model

    def train_model(self, x_train, y_train, x_val, y_val):
        fit_type = jhkaggle.jhkaggle_config['FIT_TYPE']

        if type(x_train) is not np.ndarray:
            x_train = x_train.values.astype(np.float32)
        if type(y_train) is not np.ndarray:
            y_train = y_train.values.astype(np.int32)

        if x_val is not None:
            if type(x_val) is not np.ndarray:
                x_val = x_val.values.astype(np.float32)
            if type(y_val) is not np.ndarray:
                y_val = y_val.values.astype(np.int32)

        if fit_type == jhkaggle.const.FIT_TYPE_REGRESSION:
            model = self.define_neural_network(x_train)
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')
        else:
            y_train = pd.get_dummies(y_train).values.astype(np.float32)
            if x_val is not None:
                y_val = pd.get_dummies(y_val).values.astype(np.float32)
            
            model = self.define_neural_network(x_train)
            model.add(Dense(y_train.shape[1],activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam')
            monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=0, mode='auto')
            checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", verbose=0, save_best_only=True) # save best model

        if x_val is not None:
            # Early stopping
            monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
            checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", verbose=0, save_best_only=True) # sav

            # Fit/train neural network
            model.fit(x_train,y_train,validation_data=(x_val,y_val),callbacks=[monitor,checkpointer],verbose=0,epochs=1000)
            model.load_weights('best_weights.hdf5') # load weights from best model
        else:
            model.fit(x_train,y_train,verbose=0,epochs=1000)

        #self.classifier = clr.best_iteration
        return model

    def predict_model(self, model, x):
        fit_type = jhkaggle.jhkaggle_config['FIT_TYPE']

        if type(x) is not np.ndarray:
            x = x.values.astype(np.float32)

        if fit_type == jhkaggle.const.FIT_TYPE_REGRESSION:
            pred = model.predict(x)
        else:
            pred = model.predict(x)
            pred = np.array([v[1] for v in pred])
        return pred.flatten()

    def save_model(self, path, name):
        print("Saving Model")

        self.model.save(os.path.join(path, name + ".h5"))

        meta = {
            'name': 'TrainKeras',
            'data_source': self.data_source,
            'params': self.params
        }
        
        with open(os.path.join(path,"meta.json"), 'w') as outfile:
            json.dump(meta, outfile)

    @classmethod
    def load_model(cls,path,name):
        root_path = jhkaggle.jhkaggle_config['PATH']
        model_path = os.path.join(root_path,path)
        meta_filename = os.path.join(model_path,"meta.json")
        with open(meta_filename, 'r') as fp:
            meta = json.load(fp)

        result = TrainKeras(meta['data_source'],False)
        result.model = load_model(os.path.join(model_path,name + ".h5"))
        return result



        

