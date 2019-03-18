# Jeff Heaton's Kaggle Utilities
# Copyright 2019 by Jeff Heaton, Open Source, Released under the Apache License
# For more information: https://github.com/jeffheaton/jh-kaggle-util
# 
# This is the main GLM-style ensembler.  This utility can combine multiple models.
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
import numpy as np
import os
import zipfile
import time
from sklearn.linear_model import LassoLars
from sklearn.linear_model import Lasso
import jhkaggle.util
# http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/

def fit_ensemble(x,y):
    fit_type = jhkaggle.jhkaggle_config['FIT_TYPE']
    if 1:
        if fit_type == jhkaggle.const.FIT_TYPE_BINARY_CLASSIFICATION:
            blend = SGDClassifier(loss="log", penalty="elasticnet")  # LogisticRegression()
        else:
            # blend = SGDRegressor()
            #blend = LinearRegression()
            #blend = RandomForestRegressor(n_estimators=10, n_jobs=-1, max_depth=5, criterion='mae')
            blend = LassoLarsCV(normalize=True)
            #blend = ElasticNetCV(normalize=True)
            #blend = LinearRegression(normalize=True)
        blend.fit(x, y)
    else:
        blend = LogisticRegression()
        blend.fit(x, y)


    return blend

def predict_ensemble(blend,x):
    fit_type = jhkaggle.jhkaggle_config['FIT_TYPE']

    if fit_type == jhkaggle.const.FIT_TYPE_BINARY_CLASSIFICATION:
        pred = blend.predict_proba(x)
        pred = pred[:, 1]
    else:
        pred = blend.predict(x)

    return pred

def ensemble(models):
    test_id = jhkaggle.jhkaggle_config['TEST_ID']
    target_name = jhkaggle.jhkaggle_config['TARGET_NAME']
    ensemble_oos_df = []
    ensemble_submit_df = []

    print("Loading models...")
    for model in models:
        print("Loading: {}".format(model))
        idx = model.find('-')
        suffix = model[idx:]
        path = os.path.join(jhkaggle.jhkaggle_config['PATH'], model)
        filename_oos = "oos" + suffix + ".csv"
        filename_submit = "submit" + suffix + ".csv"
        path_oos = os.path.join(path,filename_oos)
        path_submit = os.path.join(path,filename_submit)

        df_oos = pd.read_csv(path_oos)
        df_oos.sort_values(by=['id'], ascending=[1], inplace=True)
        df_submit = pd.read_csv(path_submit)
        df_submit.sort_values(by=[test_id], ascending=[1], inplace=True)

        ensemble_oos_df.append( df_oos )
        ensemble_submit_df.append( df_submit )


    ens_y = np.array(ensemble_oos_df[0]['expected'],dtype=np.int)
    ens_x = np.zeros((ensemble_oos_df[0].shape[0],len(models)))
    pred_x = np.zeros((ensemble_submit_df[0].shape[0],len(models)))

    print(ens_x.shape)

    for i, df in enumerate(ensemble_oos_df):
        ens_x[:,i] = df['predicted']

    for i, df in enumerate(ensemble_submit_df):
        pred_x[:,i] = df[target_name]

    print("Cross validating and generating OOS predictions...")

    start_time = time.time()

    x_train = jhkaggle.util.load_pandas("train-joined-1.pkl")
    folds = x_train['fold']
    num_folds = folds.nunique()
    print("Found {} folds in dataset.".format(num_folds))

    y_train = x_train['target']
    train_ids = x_train['id']
    x_train.drop('id', axis=1, inplace=True)
    x_train.drop('fold', axis=1, inplace=True)
    x_train.drop('target', axis=1, inplace=True)

    final_preds_train = np.zeros(x_train.shape[0])
    scores = []
    for fold_idx in range(num_folds):
        fold_no = fold_idx + 1
        print("*** Fold #{} ***".format(fold_no))

        # x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)
        mask_train = np.array(folds != fold_no)
        mask_test = np.array(folds == fold_no)
        fold_x_train = ens_x[mask_train]
        fold_x_valid = ens_x[mask_test]
        fold_y_train = y_train[mask_train]
        fold_y_valid = y_train[mask_test]

        print("Training data: X_train: {}, Y_train: {}, X_test: {}".format(fold_x_train.shape, len(fold_y_train),
                                                                        fold_x_valid.shape))

        fold_blend = fit_ensemble(fold_x_train,fold_y_train)
        fold_pred = predict_ensemble(fold_blend,fold_x_valid)
        score = jhkaggle.util.model_score(fold_pred,fold_y_valid)
        final_preds_train[mask_test] = fold_pred
        print("Fold score: {}".format(score))
        scores.append(score)

    score = np.mean(scores)
    print("Mean score: {}".format(score))
    print("OOS Score: {}".format(jhkaggle.util.model_score(final_preds_train,y_train)))




    print("Blending on entire dataset...")


    blend = fit_ensemble(ens_x,ens_y)

    pred = predict_ensemble(blend,pred_x)

    sub = pd.DataFrame()
    sub[test_id] = ensemble_submit_df[0][test_id]
    sub[target_name] = pred
    #stretch(sub)

    print("Writing submit file")
    #sub.to_csv(os.path.join(PATH, "glm.csv"),index=False)

    #pred2 = blend.predict_proba(ens_x)[:,1]
    #score = metrics.log_loss(ens_y, pred2)
    #print("Log loss score (training): {}".format(score))


    path, score_str, time_str = jhkaggle.util.create_submit_package("blend", score)
    filename = "submit-" + score_str + "_" + time_str
    filename_csv = os.path.join(path, filename) + ".csv"
    filename_zip = os.path.join(path, filename) + ".zip"
    filename_txt = os.path.join(path, filename) + ".txt"
    sub.to_csv(filename_csv,index=False)
    z = zipfile.ZipFile(filename_zip, 'w', zipfile.ZIP_DEFLATED)
    z.write(filename_csv, filename + ".csv")

    filename = "oos-" + score_str + "_" + time_str + ".csv"
    filename = os.path.join(path, filename)
    sub = pd.DataFrame()
    sub['id'] = train_ids
    sub['expected'] = y_train
    sub['predicted'] = final_preds_train
    sub.to_csv(filename, index=False)

    output = ""

    elapsed_time = time.time() - start_time

    output += "Elapsed time: {}\n".format(jhkaggle.util.hms_string(elapsed_time))
    output += "OOS score: {}\n".format(score)
    output += "-----Blend Results-------\n"

    z = abs(blend.coef_)
    z = z / z.sum()
    for name, d in zip(models, z):
        output += "{} : {}\n".format(d, name)


    print(output)

    with open(filename_txt, "w") as text_file:
        text_file.write(output)
