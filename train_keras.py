from util import *
import tensorflow as tf
import scipy.stats
import numpy as np
import time
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint


class TrainTensorFlow(TrainModel):
    def __init__(self, data_source, params, run_single_fold):
        super().__init__(data_source, run_single_fold)
        self.name="tensorflow"
        self.params = params
        self.early_stop = 50


    def train_model(self, x_train, y_train, x_val, y_val):

        if type(x_train) is not np.ndarray:
            x_train = x_train.values.astype(np.float32)
        if type(y_train) is not np.ndarray:
            y_train = y_train.values.astype(np.int32)

        if x_val is not None:
            if type(x_val) is not np.ndarray:
                x_val = x_val.values.astype(np.float32)
            if type(y_val) is not np.ndarray:
                y_val = y_val.values.astype(np.int32)

        if FIT_TYPE == FIT_TYPE_REGRESSION:
            model = Sequential()
            model.add(Dense(20, input_dim=x_train.shape[1], activation='relu'))
            model.add(Dense(10, activation='relu'))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')
        else:
            model = Sequential()
            model.add(Dense(20, input_dim=x.shape[1], activation='relu'))
            model.add(Dense(10))
            model.add(Dense(y.shape[1],activation='softmax'))
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
        if type(x) is not np.ndarray:
            x = x.values.astype(np.float32)

        if FIT_TYPE == FIT_TYPE_REGRESSION:
            pred = model.predict(x)
        else:
            pred = model.predict(x)
            pred = np.array([v[1] for v in pred])
        return pred.flatten()

# "all the time" to "always"
# reall short ones that are dead wrong

# [100]	train-logloss:0.288795	eval-logloss:0.329036
# [598]	train-logloss:0.152968	eval-logloss:0.296854
# [984]	train-logloss:0.096444	eval-logloss:0.293915

tf.logging.set_verbosity(tf.logging.INFO)
params = {
    'opt':tf.train.AdamOptimizer(learning_rate=1e-3),
    'hidden':[500,100,50],
    'seed':42,
    'dropout': 0.2 # probability of dropping out a given neuron
}
start_time = time.time()
train = TrainTensorFlow("1",params,False)
train.zscore = False
train.run()

elapsed_time = time.time() - start_time
print("Elapsed time: {}".format(hms_string(elapsed_time)))

