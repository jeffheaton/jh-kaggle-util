# Jeff Heaton's Kaggle Utilities
# Copyright 2019 by Jeff Heaton, Open Source, Released under the Apache License
# For more information: https://github.com/jeffheaton/jh-kaggle-util
# 
# Tune for XGBoost, based on: https://towardsdatascience.com/fine-tuning-xgboost-in-python-like-a-boss-b4543ed8b1e
#

from util import *
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
import multiprocessing
import itertools
import demjson
import time

import pandas as pd
import numpy as np
import xgboost as xgb
import time
import os
import zipfile
import operator
from sklearn.metrics import log_loss
import scipy
from train_xgboost import TrainXGBoost

# http://stackoverflow.com/questions/2853212/all-possible-permutations-of-a-set-of-lists-in-python

FOLDS = 5
EARLY_STOP = 50
MAX_ROUNDS = 5

PARAMS1 = {
'objective': 'reg:linear',
    'eval_metric': 'rmse',
'silent' : 1,
    'learning_rate':0.0045,'seed':4242
}

train = TrainXGBoost("1",params=PARAMS1,run_single_fold=True)

def modelfit(params,x,y):
    #Fit the algorithm on the data
    print("fit")
    alg = XGBClassifier(**params)
    alg.fit(x,y,verbose=True)
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    print(feat_imp)

def update(base_dict, update_copy):
    for key in update_copy.keys():
        base_dict[key] = update_copy[key]

def step1_find_depth_and_child(params):
    test1 = {
        'max_depth': list(range(3,12,2)),
        'min_child_weight': list(range(1,10,2))
    }

    return grid_search(params, test1, 1)

def step2_narrow_depth(params):
    max_depth = params['max_depth']
    test2 = {
        'max_depth': [max_depth-1,max_depth,max_depth+1]
    }
    return grid_search(params, test2, 2)

def step3_gamma(params):
    test3 = {
        'gamma': list([i/10.0 for i in range(0,5)])
    }
    return grid_search(params, test3, 3)

def step4_sample(params):
    test4 = {
        'subsample':list([i/10.0 for i in range(6,10)]),
        'colsample_bytree':list([i/10.0 for i in range(6,10)])
    }
    return grid_search(params, test4, 4)

def step5_reg1(params):
    test5 = {
        'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
    }
    return grid_search(params, test5, 5)




def grid_search(params,grid,num):
    keys = set(grid.keys())
    l = [grid[x] for x in keys]
    perm = list(itertools.product(*l))
    jobs = []
    for i in perm:
        jobs.append({k:v for k,v in zip(keys,i)})

    print("Total number of jobs: {}".format(len(jobs)))
    column_step = []
    column_score = []
    column_jobs = []

    for i,job in enumerate(jobs):
        print("** Starting job: {}:{}/{}".format(num,i+1,len(jobs)))
        params2 = dict(params)
        update(params2,job)
        train.params = params2
        train.rounds = MAX_ROUNDS
        train.early_stop = EARLY_STOP
        result = train.run_cv()
        print("Result: {}".format(result))
        column_jobs.append(str(job))
        column_score.append(result[0])
        column_step.append(result[1])

    df = pd.DataFrame({'job':column_jobs,'step':column_step,'score':column_score},columns=['job','score','step'])
    df.sort_values(by=['score'],ascending=[True],inplace=True)
    print(df)
    path_tune = os.path.join(PATH, "tune-{}.csv".format(num))
    df.to_csv(path_tune, index=False)
    j = df.iloc[0]['job']
    j = demjson.decode(j)
    return j


def main():
    print("loading")
    start_time = time.time()
    train._run_startup()
    params = dict(PARAMS1)

    # Step 1
    step1_result = step1_find_depth_and_child(params)
    #step1_result = {'max_depth': 3, 'min_child_weight': 7}
    update(params,step1_result)
    print("Step 1 Result: {}".format(step1_result))

    # Step 2
    step2_result = step2_narrow_depth(params)
    #step2_result = {'max_depth': 4}
    update(params, step2_result)
    print("Step 2 Result: {}".format(step2_result))

    # Step 3
    step3_result = step3_gamma(params)
    #step3_result = {'gamma': 0.4}
    update(params, step3_result)
    print("Step 3 Result: {}".format(step3_result))

    # Step 4
    step4_result = step4_sample(params)
    #step4_result = {'colsample_bytree': 0.9, 'subsample': 0.8}
    update(params, step4_result)
    print("Step 4 Result: {}".format(step4_result))

    # Step 5
    step5_result = step5_reg1(params)
    #step5_result = {'colsample_bytree': 0.9, 'subsample': 0.8}
    update(params, step5_result)
    print("Step 5 Result: {}".format(step5_result))

    print("Final Result: {}".format(params))
    elapsed_time = time.time() - start_time
    print("Elapsed time: {}".format(hms_string(elapsed_time)))





if __name__ == '__main__':
    main()
