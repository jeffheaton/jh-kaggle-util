# Jeff Heaton's Kaggle Utilities
# Copyright 2019 by Jeff Heaton, Open Source, Released under the Apache License
# For more information: https://github.com/jeffheaton/jh-kaggle-util
# 
# Utility functions.
#
import jhkaggle.const
import codecs
import math
import csv
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
import shutil
import os
import time
from tqdm import tqdm
from sklearn.metrics import log_loss,auc,roc_curve
import zipfile
import scipy
import json
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils import check_array

NA_VALUES = ['NA', '?', '-inf', '+inf', 'inf', '', 'nan']

def save_pandas(df,filename):
    path = jhkaggle.jhkaggle_config['PATH']
    if filename.endswith(".csv"):
        df.to_csv(os.path.join(path, filename),index=False)
    elif filename.endswith(".pkl"):
        df.to_pickle(os.path.join(path, filename))

def load_pandas(filename):
    path = jhkaggle.jhkaggle_config['PATH']
    if filename.endswith(".csv"):
        return pd.read_csv(os.path.join(path, filename), na_values=NA_VALUES)
    elif filename.endswith(".pkl"):
        return pd.read_pickle(os.path.join(path, filename))


def balance(df_train):
    y_train = df_train['target'].values
    pos_train = df_train[y_train == 1]
    neg_train = df_train[y_train == 0]

    orig_ser = pos_train['id'].append(neg_train['id'])

    max_id = int(max(df_train['id']))
    balance_id = math.ceil(max_id / 100000) * 100000
    print("First balanced ID: {}".format(balance_id))

    print("Oversampling started for proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))
    print("Before oversample: pos={},neg={}".format(len(pos_train), len(neg_train)))
    p = 0.17426  # 0.165
    scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
    current_neg_len = len(neg_train)
    while scale > 1:
        neg_train = pd.concat([neg_train, neg_train])
        scale -= 1
    neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
    added = len(neg_train) - current_neg_len

    print("Oversampling done, new proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))
    df_train = pd.concat([pos_train, neg_train])
    print("After oversample: pos={},neg={}".format(len(pos_train), len(neg_train)))

    # Fill in id's for new data
    added_ser = pd.Series(range(balance_id, balance_id + added))
    new_id = orig_ser.append(added_ser)
    df_train['id'] = new_id.values

    del pos_train, neg_train

    df_train.sort_values(by=["id"], inplace=True)
    df_train.reset_index(inplace=True, drop=True)
    return df_train


# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


# Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1] for red,green,blue)
def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)


# Encode text values to a single dummy variable.  The new columns (which do not replace the old) will have a 1
# at every location where the original column (name) matches each of the target_values.  One column is added for
# each target value.
def encode_text_single_dummy(df, name, target_values):
    for tv in target_values:
        l = list(df[name].astype(str))
        l = [1 if str(x) == str(tv) else 0 for x in l]
        name2 = "{}-{}".format(name, tv)
        df[name2] = l


# Encode text values to indexes(i.e. [1],[2],[3] for red,green,blue).
def encode_text_index(df, name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_


# Encode a numeric column as zscores
def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd


# Convert all missing values in the specified column to the median
def missing_median(df, name):
    med = df[name].median()
    df[name] = df[name].fillna(med)


# Convert all missing values in the specified column to the default
def missing_default(df, name, default_value):
    df[name] = df[name].fillna(default_value)


# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)

    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(target_type, '__iter__') else target_type

    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        return df.as_matrix(result).astype(np.float32), df.as_matrix([target]).astype(np.int32)
    else:
        # Regression
        return df.as_matrix(result).astype(np.float32), df.as_matrix([target]).astype(np.float32)

# Regression chart, we will see more of this chart in the next class.
def chart_regression(pred, y):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    t.sort_values(by=['y'], inplace=True)
    a = plt.plot(t['y'].tolist(), label='expected')
    b = plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()


# Get a new directory to hold checkpoints from a neural network.  This allows the neural network to be
# loaded later.  If the erase param is set to true, the contents of the directory will be cleared.
def get_model_dir(name, erase):
    path = jhkaggle.jhkaggle_config['PATH']
    model_dir = os.path.join(path, name)
    os.makedirs(model_dir, exist_ok=True)
    if erase and len(model_dir) > 4 and os.path.isdir(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)  # be careful, this deletes everything below the specified path
    return model_dir


# Remove all rows where the specified column is +/- sd standard deviations
def remove_outliers(df, name, sd):
    drop_rows = df.index[(np.abs(df[name] - df[name].mean()) >= (sd * df[name].std()))]
    df.drop(drop_rows, axis=0, inplace=True)


# Encode a column to a range between normalized_low and normalized_high.
def encode_numeric_range(df, name, normalized_low=-1, normalized_high=1,
                         data_low=None, data_high=None):
    if data_low is None:
        data_low = min(df[name])
        data_high = max(df[name])

    df[name] = ((df[name] - data_low) / (data_high - data_low)) \
               * (normalized_high - normalized_low) + normalized_low


def create_submit_package(name, score):
    score_str = str(round(float(score), 6)).replace('.', 'p')
    time_str = time.strftime("%Y%m%d-%H%M%S")
    filename = name + "-" + score_str + "_" + time_str
    path = os.path.join(jhkaggle.jhkaggle_config['PATH'], filename)
    if not os.path.exists(path):
        os.makedirs(path)
    return path, score_str, time_str


def stretch(y):
    return (y - y.min()) / (y.max() - y.min())

def model_score(y_pred,y_valid):
    final_eval = jhkaggle.jhkaggle_config['FINAL_EVAL']
    if final_eval == jhkaggle.const.EVAL_R2:
        return r2_score(y_valid, y_pred)
    elif final_eval == jhkaggle.const.EVAL_LOGLOSS:
        return log_loss(y_valid, y_pred)
    elif final_eval == jhkaggle.const.EVAL_AUC:
        fpr, tpr, thresholds = roc_curve(y_valid, y_pred, pos_label=1)
        return auc(fpr, tpr)
    else:
        raise Exception(f"Unknown FINAL_EVAL: {final_eval}")



class TrainModel:
    def __init__(self, data_source, run_single_fold):
        self.data_source = data_source
        self.run_single_fold = run_single_fold
        self.num_folds = None
        self.zscore = False
        self.steps = None # How many steps to the best model
        self.cv_steps = [] # How many steps at each CV fold
        self.rounds = None # How many rounds are desired (if supported by model)
        self.pred_denom = 1

    def _run_startup(self):
        self.start_time = time.time()
        self.x_train = load_pandas("train-joined-{}.pkl".format(self.data_source))
        self.x_submit = load_pandas("test-joined-{}.pkl".format(self.data_source))

        self.input_columns = list(self.x_train.columns.values)

        # Grab what columns we need, but are not used for training
        self.train_ids = self.x_train['id']
        self.y_train = self.x_train['target']
        self.submit_ids = self.x_submit['id']
        self.folds = self.x_train['fold']
        self.num_folds = self.folds.nunique()
        print("Found {} folds in dataset.".format(self.num_folds))

        # Drop what is not used for training
        self.x_train.drop('id', axis=1, inplace=True)
        self.x_train.drop('fold', axis=1, inplace=True)
        self.x_train.drop('target', axis=1, inplace=True)
        self.x_submit.drop('id', axis=1, inplace=True)

        self.input_columns2 = list(self.x_train.columns.values)
        self.final_preds_train = np.zeros(self.x_train.shape[0])
        self.final_preds_submit = np.zeros(self.x_submit.shape[0])

        for i in range(len(self.x_train.dtypes)):
            dt = self.x_train.dtypes[i]
            name = self.x_train.columns[i]

            if dt not in [np.float64, np.float32, np.int32, np.int64]:
                print("Bad type: {}:{}".format(name,name.dtype))

            elif self.x_train[name].isnull().any():
                print("Null values: {}".format(name))

        if self.zscore:
            self.x_train = scipy.stats.zscore(self.x_train)
            self.x_submit = scipy.stats.zscore(self.x_submit)

    def _run_cv(self):
        folds2run = self.num_folds if not self.run_single_fold else 1

        for fold_idx in range(folds2run):
            fold_no = fold_idx + 1
            print("*** Fold #{} ***".format(fold_no))

            # x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)
            mask_train = np.array(self.folds != fold_no)
            mask_test = np.array(self.folds == fold_no)
            fold_x_train = self.x_train[mask_train]
            fold_x_valid = self.x_train[mask_test]
            fold_y_train = self.y_train[mask_train]
            fold_y_valid = self.y_train[mask_test]

            print("Training data: X_train: {}, Y_train: {}, X_test: {}".format(fold_x_train.shape, len(fold_y_train),
                                                                               self.x_submit.shape))
            self.model = self.train_model(fold_x_train, fold_y_train, fold_x_valid, fold_y_valid)
            preds_valid = self.predict_model(self.model, fold_x_valid)

            score = model_score(preds_valid,fold_y_valid)

            preds_submit = self.predict_model(self.model, self.x_submit)

            self.final_preds_train[mask_test] = preds_valid
            self.final_preds_submit += preds_submit
            self.denom += 1
            self.pred_denom +=1

            if self.steps is not None:
                self.cv_steps.append(self.steps)

            self.scores.append(score)
            print("Fold score: {}".format(score))

            if fold_no==1:
                self.model_fold1 = self.model
        self.score = np.mean(self.scores)

        if len(self.cv_steps)>0:
            self.rounds = max(self.cv_steps) # Choose how many rounds to use after all CV steps

    def _run_single(self):
        print("Training data: X_train: {}, Y_train: {}, X_test: {}".format(self.x_train.shape, len(self.y_train),
                                                                               self.x_submit.shape))
        self.model = self.train_model(self.x_train, self.y_train, None, None)

#        if not self.run_single_fold:
#            self.preds_oos = self.predict_model(self.model, self.x_train)

        #score = 0 #log_loss(fold_y_valid, self.preds_oos)

        #self.final_preds_train = self.preds_oos
        self.final_preds_submit = self.predict_model(self.model, self.x_submit)
        self.pred_denom = 1

    def _run_assemble(self):
        target_name = jhkaggle.jhkaggle_config['TARGET_NAME']
        test_id = jhkaggle.jhkaggle_config['TEST_ID']

        print("Training done, generating submission file")

        if len(self.scores)==0:
            self.denom = 1
            self.scores.append(-1)
            self.score = -1
            print("Warning, could not produce a validation score.")


        # create folder
        path, score_str, time_str = create_submit_package(self.name, self.score)

        filename = "submit-" + score_str + "_" + time_str
        filename_csv = os.path.join(path, filename) + ".csv"
        filename_zip = os.path.join(path, filename) + ".zip"
        filename_txt = os.path.join(path, filename) + ".txt"

        sub = pd.DataFrame()
        sub[test_id] = self.submit_ids
        sub[target_name] = self.final_preds_submit / self.pred_denom
        print("Pred denom: {}".format(self.pred_denom))
        sub.to_csv(filename_csv, index=False)

        z = zipfile.ZipFile(filename_zip, 'w', zipfile.ZIP_DEFLATED)
        z.write(filename_csv, filename + ".csv")
        output = ""
        # Generate training OOS file
        if not self.run_single_fold:
            filename = "oos-" + score_str + "_" + time_str + ".csv"
            filename = os.path.join(path, filename)
            sub = pd.DataFrame()
            sub['id'] = self.train_ids
            sub['expected'] = self.y_train
            sub['predicted'] = self.final_preds_train
            sub.to_csv(filename, index=False)
            output+="OOS Score: {}".format(model_score(self.final_preds_train,self.y_train))
            self.save_model(path, 'model-submit')
            if self.model_fold1:
                t = self.model
                self.model = self.model_fold1
                self.save_model(path, 'model-fold1')
                self.model = t

        print("Generated: {}".format(path))
        elapsed_time = time.time() - self.start_time

        output += "Elapsed time: {}\n".format(hms_string(elapsed_time))
        output += "Mean score: {}\n".format(self.score)
        output += "Fold scores: {}\n".format(self.scores)
        output += "Params: {}\n".format(self.params)
        output += "Columns: {}\n".format(self.input_columns)
        output += "Columns Used: {}\n".format(self.input_columns2)
        output += "Steps: {}\n".format(self.steps)

        output += "*** Model Specific Feature Importance ***\n"
        output = self.feature_rank(output)

        print(output)

        with open(filename_txt, "w") as text_file:
            text_file.write(output)

    def feature_rank(self,output):
        return output

    def run(self):
        self.denom = 0
        self.scores = []

        self._run_startup()
        self._run_cv()
        print("Fitting single model for entire training set.")
        self._run_single()
        self._run_assemble()

    def save_model(self, path, name):
        print("Saving Model")
        with open(os.path.join(path, name + ".pkl"), 'wb') as fp:  
            pickle.dump(self.model, fp)

        meta = {
            'name': self.__class__.__name__,
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

        result = cls(meta['data_source'],meta['params'],False)
        with open(os.path.join(root, name + ".pkl"), 'rb') as fp:  
            result.model = pickle.load(fp)
        return result

class GenerateDataFile:
    def __init__(self, name, columns):
        self.name = name
        self.columns = columns
        self.max_lines = None
        self.stats_pos = {}
        self.stats_neg = {}

        #for col in self.columns:
        #    self.stats_neg[col.name] = []
        #    self.stats_pos[col.name] = []

    def preprocess_needed(self):
        for col in self.columns:
            pp = getattr(col, "preprocess", None)
            if callable(pp):
                return True
        return False

    def preprocess(self,ifile):
        target_name = jhkaggle.jhkaggle_config['TARGET_NAME']
        
        header_idx = {key: value for (value, key) in enumerate(next(ifile))}

        for row in tqdm(ifile):
            if target_name in header_idx:
                target = row[header_idx[target_name]]
            else:
                target = None

            for col in self.columns:
                pp = getattr(col, "preprocess", None)
                if callable(pp):
                    col.preprocess(header_idx,row)

    def notify_begin(self):
        for col in self.columns:
            bg = getattr(col, "begin", None)
            if callable(bg):
                col.begin()


    def process(self, task, ifile, ofile):
        global global_target
        target_name = jhkaggle.jhkaggle_config['TARGET_NAME']
        fit_type = jhkaggle.jhkaggle_config['FIT_TYPE']
        print("Process: {}".format(task))

        header_idx = {key: value for (value, key) in enumerate(next(ifile))}

        i = 0
        for row in tqdm(ifile):
            id = row[0]

            if target_name in header_idx:
                target = row[header_idx[target_name]]
            else:
                target = None

            values = []
            for col in self.columns:
                values += col.process(header_idx,row)

            row2 = [id]
            if target is not None:
                global_target = float(target)
                ofile.writerow(row2 + values + [target])

                #if fit_type == jhkaggle.const.FIT_TYPE_BINARY_CLASSIFICATION:
                #    self.track_binary(values,target)


            else:
                global_target = -1
                ofile.writerow(row2 + values)

            i+=1
            if self.max_lines is not None and i>self.max_lines:
                break

    def track_binary(self,values,target):
        for col, value in zip(self.columns, values):
            if int(target) == 1:
                self.stats_pos[col.name].append(float(value))
            else:
                self.stats_neg[col.name].append(float(value))

    def report(self):
        fit_type = jhkaggle.jhkaggle_config['FIT_TYPE']

#        if fit_type == jhkaggle.const.FIT_TYPE_BINARY_CLASSIFICATION:
#            for col in self.columns:
#                p = self.stats_pos[col.name]
#                n = self.stats_neg[col.name]
#                mean_p = sum(p)/float(len(p))
#                mean_n = sum(n)/float(len(n))
#                print("{}:pos={},neg={},diff={}".format(col.name,mean_p,mean_n,abs(mean_p-mean_n)))


    def run(self):
        path = jhkaggle.jhkaggle_config['PATH']
        encoding = jhkaggle.jhkaggle_config['ENCODING']

        filename_read_train = os.path.join(path, "train.csv")
        filename_read_test = os.path.join(path, "test.csv")

        filename_write_train = os.path.join(path, "data-{}-train.csv".format(self.name))
        filename_write_test = os.path.join(path, "data-{}-test.csv".format(self.name))

        filename_write_train2 = os.path.join(path, "data-{}-train.pkl".format(self.name))
        filename_write_test2 = os.path.join(path, "data-{}-test.pkl".format(self.name))

        start_time = time.time()

        if self.preprocess_needed():
            with codecs.open(filename_read_train, "r", encoding) as fhr1:
                reader_train = csv.reader(fhr1)
                self.preprocess(reader_train)

            with codecs.open(filename_read_test, "r", encoding) as fhr2:
                reader_test = csv.reader(fhr2)
                self.preprocess(reader_test)

        self.notify_begin()

        with codecs.open(filename_read_train, "r", encoding) as fhr1, \
                codecs.open(filename_read_test, "r", encoding) as fhr2, \
                codecs.open(filename_write_train, "w", encoding) as fhw1, \
                codecs.open(filename_write_test, "w", encoding) as fhw2:
            reader_train = csv.reader(fhr1)
            reader_test = csv.reader(fhr2)
            writer_train = csv.writer(fhw1)
            writer_test = csv.writer(fhw2)

            column_names = []
            for col in self.columns:
                column_names += col.name

            writer_train.writerow(['id'] + column_names + ['target'])
            writer_test.writerow(['id'] + column_names)

            self.process("train", reader_train, writer_train)
            self.process("test", reader_test, writer_test)

        elapsed_time = time.time() - start_time
        print("Converting to PKL")
        df = load_pandas(filename_write_train)
        save_pandas(df,filename_write_train2)

        df = load_pandas(filename_write_test)
        save_pandas(df, filename_write_test2)

        print("Elapsed time: {}".format(hms_string(elapsed_time)))



class OneHot:
    def __init__(self,col_name):
        self.col_name = col_name
        self.name = None
        self.values = set()
        self.value_idx = {}

    def preprocess(self, header_idx, row):
        self.values.add(row[header_idx[self.col_name]])

    def begin(self):
        self.values = list(self.values)
        self.values.sort()
        self.name = ["{}-{}".format(self.col_name,x) for x in self.values]
        self.value_idx = {key: value for (value, key) in enumerate(self.values)}

    def process(self, header_idx, row):
        value = row[header_idx[self.col_name]]
        result = [0] * len(self.values)
        result[self.value_idx[value]] = 1
        return result


class PassThru:
    def __init__(self, col_names):
        self.name = col_names

    def process(self, header_idx, row):
        result = []
        for n in self.name:
            result.append(row[header_idx[n]])
        return result

class CatMeanTarget:
    def __init__(self,target_name,col_name):
        self.col_name = col_name
        self.target_name = target_name
        self.name = ["mean-tgt:{}".format(col_name)]
        self.value_count = {}
        self.value_sum = {}
        self.value_mean = {}

    def preprocess(self, header_idx, row):
        value = row[header_idx[self.col_name]]

        if self.target_name not in header_idx:
            return

        target = row[header_idx[self.target_name]]

        self.value_count[value] = self.value_count.get(value,0) + 1
        self.value_sum[value] = self.value_sum.get(value, 0) + float(target)

    def begin(self):
        for key in self.value_count.keys():
            self.value_mean[key] = self.value_sum[key] / self.value_count[key]

        self.overall_mean = np.mean(np.array(list(self.value_mean.values())))

    def process(self, header_idx, row):
        value = row[header_idx[self.col_name]]
        if value not in self.value_mean:
            return [self.overall_mean]
        else:
            return [self.value_mean[value]]

class SumColumns:
    def __init__(self, name, col_names):
        self.name = [ "sum:".format(name) ]
        self.col_names = col_names

    def process(self, header_idx, row):
        sum = 0
        count = 0
        for col in self.col_names:
            value = row[header_idx[col]]
            count+=1
            sum+=float(value)

        return [sum]


class StackingEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self

    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prodiction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed

def load_model(model_name):
    idx = model_name.find('-')
    suffix = model_name[idx:]
    path = os.path.join( jhkaggle.jhkaggle_config['PATH'], model_name)

    filename_oos = "oos" + suffix + ".csv"
    path_oos = os.path.join(path, filename_oos)
    df_oos = pd.read_csv(path_oos)

    filename_submit = "submit" + suffix + ".csv"
    path_submit = os.path.join(path, filename_submit)
    df_submit = pd.read_csv(path_submit)

    return df_oos, df_submit

def save_importance_report(model,imp):
  root_path = jhkaggle.jhkaggle_config['PATH']
  model_path = os.path.join(root_path,model)
  imp.to_csv(os.path.join(model_path,'peturb.csv'),index=False)

