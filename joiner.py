# Jeff Heaton's Kaggle Utilities
# Copyright 2019 by Jeff Heaton, Open Source, Released under the Apache License
# For more information: https://github.com/jeffheaton/jh-kaggle-util
# 
# The joiner combines the features from the data- files and produces the feature set to train on.
import pandas as pd
import os
import os.path
from config import *
from util import *
from sklearn.model_selection import KFold

PROFILE_1 = {
    'name' : '1',
    'SOURCES' : ['jth-1','ker1'],
    'BALANCE' : False,
    'IGNORE' : [],
    'ORIG_FIELDS' : []
}

FOLDS = 10
SEED = 42

def perform_join(profile):
    df_train_joined = None
    df_test_joined = None
    data_columns = []

    if len(profile['ORIG_FIELDS']) > 0:
        df_train_orig = pd.read_csv(os.path.join(PATH, "train.csv"), na_values=NA_VALUES)
        df_test_orig = pd.read_csv(os.path.join(PATH, "test.csv"), na_values=NA_VALUES)

    for source in profile['SOURCES']:
        print("Processing: {}".format(source))
        filename_train = "data-{}-train.pkl".format(source)
        filename_test = "data-{}-test.pkl".format(source)

        if not os.path.exists(filename_train):
            filename_train = "data-{}-train.csv".format(source)
            filename_test = "data-{}-test.csv".format(source)
            
        df_train = load_pandas(filename_train)
        df_test = load_pandas(filename_test)

        df_train.sort_values(by=['id'], ascending=[1], inplace=True)
        df_test.sort_values(by=['id'], ascending=[1], inplace=True)

        #if df_train.shape[0] != df_test.shape[0]:
        #    raise Exception("Row count of {} and {} do not match.".format(filename_test,filename_train))

        if df_train_joined is None:
            df_train_joined = pd.DataFrame()
            df_test_joined = pd.DataFrame()

            df_train_joined['id'] = df_train['id']
            df_test_joined['id'] = df_test['id']

        # Copy columns

        feature_names = list(df_train.columns.values)

        for name in feature_names:
            col_name = "{}:{}".format(source, name)
            if name == 'id' or name == 'fold' or name == 'target' or col_name in profile['IGNORE']:
                continue
            data_columns.append(col_name)
            df_train_joined[col_name] = df_train[name]
            df_test_joined[col_name] = df_test[name]

    # Eliminate any missing values
    print("Missing")
    for name in data_columns:
        med = df_train_joined[name].median()
        df_train_joined[name] = df_train_joined[name].fillna(med)
        df_test_joined[name] = df_test_joined[name].fillna(med)

    # Add in any requested orig fields
    print("Orig fields")
    for name in profile['ORIG_FIELDS']:
        col_name = "{}:{}".format('orig', name)
        df_train_joined[name] = df_train_orig[name]
        df_test_joined[name] = df_test_orig[name]

    # Add target
    print("Target")
    df_train_joined['target'] = df_train['target'] # get the target from the last file joined (targets SHOULD be all the same)

    # Balance
    if profile['BALANCE']:  # Now we oversample the negative class - on your own risk of overfitting!
        print("Balance")
        df_train_joined = balance(df_train_joined)

    # Designate folds
    print("Folding")
    df_train_joined.insert(1, 'fold', 0)
    kf = KFold(FOLDS, shuffle=True, random_state=SEED)
    fold = 1
    fold = 1
    for train, test in kf.split(df_train_joined):
        df_train_joined.ix[test, 'fold'] = fold
        fold += 1

    # Write joined files
    print("Writing output...")
    save_pandas(df_train_joined,"train-joined-{}.pkl".format(profile['name']))
    save_pandas(df_test_joined,"test-joined-{}.pkl".format(profile['name']))

perform_join(PROFILE_1)