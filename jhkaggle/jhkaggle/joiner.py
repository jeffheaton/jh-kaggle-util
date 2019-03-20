# Jeff Heaton's Kaggle Utilities
# Copyright 2019 by Jeff Heaton, Open Source, Released under the Apache License
# For more information: https://github.com/jeffheaton/jh-kaggle-util
# 
# The joiner combines the features from the data- files and produces the feature set to train on.
import pandas as pd
import os
import os.path
import jhkaggle
import jhkaggle.util
from sklearn.model_selection import KFold,StratifiedKFold

def perform_join(profile_name):
    path = jhkaggle.jhkaggle_config['PATH']
    fit_type = jhkaggle.jhkaggle_config['FIT_TYPE']
    target_name = jhkaggle.jhkaggle_config['TARGET_NAME']

    df_train_joined = None
    df_test_joined = None
    data_columns = []

    if profile_name not in jhkaggle.jhkaggle_config['JOIN_PROFILES']:
        raise Error(f"Undefined join profile: {profile_name}")
    profile = jhkaggle.jhkaggle_config['JOIN_PROFILES'][profile_name]
    folds = jhkaggle.jhkaggle_config['FOLDS']
    seed = jhkaggle.jhkaggle_config['SEED']

    if len(profile['ORIG_FIELDS']) > 0:
        df_train_orig = pd.read_csv(os.path.join(path, "train.csv"), na_values=NA_VALUES)
        df_test_orig = pd.read_csv(os.path.join(path, "test.csv"), na_values=NA_VALUES)

    for source in profile['SOURCES']:
        print("Processing: {}".format(source))
        filename_train = "data-{}-train.pkl".format(source)
        filename_test = "data-{}-test.pkl".format(source)

        if not os.path.exists(os.path.join(path, filename_train)):
            filename_train = "data-{}-train.csv".format(source)
            filename_test = "data-{}-test.csv".format(source)
            
        df_train = jhkaggle.util.load_pandas(filename_train)
        df_test = jhkaggle.util.load_pandas(filename_test)
        #df_train = df_train[0:10000]

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

    fold = 1
    if fit_type == jhkaggle.const.FIT_TYPE_REGRESSION:
        kf = KFold(folds, shuffle=True, random_state=seed)
        for train, test in kf.split(df_train_joined):
            df_train_joined.ix[test, 'fold'] = fold
            fold += 1
    else:
        targets = df_train_joined[target_name]
        kf = StratifiedKFold(folds, shuffle=True, random_state=seed)
        for train, test in kf.split(df_train_joined, targets):
            df_train_joined.ix[test, 'fold'] = fold
            fold += 1

    # Write joined files
    print("Writing output...")
    jhkaggle.util.save_pandas(df_train_joined,f"train-joined-{profile_name}.pkl")
    jhkaggle.util.save_pandas(df_test_joined,f"test-joined-{profile_name}.pkl")
    jhkaggle.util.save_pandas(df_train_joined,f"train-joined-{profile_name}.csv")
    jhkaggle.util.save_pandas(df_test_joined,f"test-joined-{profile_name}.csv")
