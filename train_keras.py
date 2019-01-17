from util import *
import tensorflow as tf
import tensorflow.contrib.learn as learn
import scipy.stats
import numpy as np
import time

class TrainTensorFlow(TrainModel):
    def __init__(self, data_source, params, run_single_fold):
        super().__init__(data_source, run_single_fold)
        self.name="tensorflow"
        self.params = params
        self.early_stop = 50


    def train_model(self, x_train, y_train, x_val, y_val):


        print(type(x_train))
        if type(x_train) is not np.ndarray:
            x_train = x_train.as_matrix().astype(np.float32)
        if type(y_train) is not np.ndarray:
            y_train = y_train.as_matrix().astype(np.int32)

        if x_val is not None:
            if type(x_val) is not np.ndarray:
                x_val = x_val.as_matrix().astype(np.float32)
            if type(y_val) is not np.ndarray:
                y_val = y_val.as_matrix().astype(np.int32)


        # Get/clear a directory to store the neural network to
        model_dir = get_model_dir('dnn_kaggle',True)

        # Create a deep neural network
        feature_columns = [tf.contrib.layers.real_valued_column("", dimension=x_train.shape[1])]

        if FIT_TYPE == FIT_TYPE_REGRESSION:
            classifier = learn.DNNRegressor(
                optimizer=self.params['opt'],
                dropout=self.params['dropout'],
                model_dir=model_dir,
                config=tf.contrib.learn.RunConfig(save_checkpoints_secs=60),
                hidden_units=self.params['hidden'], feature_columns=feature_columns)
        else:
            classifier = learn.DNNClassifier(
                optimizer=self.params['opt'],
                dropout=self.params['dropout'],
                model_dir= model_dir,
                config=tf.contrib.learn.RunConfig(save_checkpoints_secs=60),
                hidden_units=self.params['hidden'], n_classes=self.params['n_classes'], feature_columns=feature_columns)

        if x_val is not None:
            # Early stopping
            validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
                x_val,
                y_val,
                every_n_steps=100,
                #metrics=validation_metrics,
                early_stopping_metric="loss",
                early_stopping_metric_minimize=True,
                early_stopping_rounds=500)

            # Fit/train neural network
            classifier.fit(x_train, y_train,monitors=[validation_monitor],steps=100000, batch_size=1000)
            self.steps = validation_monitor._best_value_step
        else:
            classifier.fit(x_train, y_train, steps=100000, batch_size=self.rounds)

        #self.classifier = clr.best_iteration
        return classifier

    def predict_model(self, clr, x):
        if type(x) is not np.ndarray:
            x = x.as_matrix().astype(np.float32)

        if FIT_TYPE == FIT_TYPE_REGRESSION:
            pred = np.array(list(clr.predict(x, as_iterable=True)))
        else:
            pred = list(clr.predict_proba(x, as_iterable=True))
            pred = np.array([v[1] for v in pred])
        return pred

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
train = TrainTensorFlow("2",params,False)
train.zscore = False
train.run()

elapsed_time = time.time() - start_time
print("Elapsed time: {}".format(hms_string(elapsed_time)))

