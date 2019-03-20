# Jeff Heaton's Kaggle Utilities
# Copyright 2019 by Jeff Heaton, Open Source, Released under the Apache License
# For more information: https://github.com/jeffheaton/jh-kaggle-util
# 
# The data- files are used to generate features from the raw data files provided by Kaggle.
# These features individual feature files are then joined together (by joiner.py) to become
# the feature vector that training actually occurs on. 
#
# Based on code found here:
# https://www.kaggle.com/tobikaggle/stacked-then-averaged-models-0-5697
import config
import jhkaggle
import jhkaggle.util
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection

path = jhkaggle.jhkaggle_config['PATH']
encoding = jhkaggle.jhkaggle_config['ENCODING']

NAME = "reduce"

# read source files
print("Loading Kaggle source files...")
filename_read_train = os.path.join(path, "train.csv")
filename_read_test = os.path.join(path, "test.csv")

train_df = pd.read_csv(filename_read_train,encoding=encoding)
test_df = pd.read_csv(filename_read_test,encoding=encoding)

train_id = train_df[jhkaggle.jhkaggle_config['TRAIN_ID']]
test_id = test_df[jhkaggle.jhkaggle_config['TEST_ID']]
train_target = train_df[jhkaggle.jhkaggle_config['TARGET_NAME']]
train_df.drop( jhkaggle.jhkaggle_config['TRAIN_ID'],axis=1,inplace=True)
train_df.drop( jhkaggle.jhkaggle_config['TARGET_NAME'],axis=1,inplace=True)
test_df.drop( jhkaggle.jhkaggle_config['TEST_ID'],axis=1,inplace=True)

################
# transform
###############
print("Transforming...")
n_comp = 50

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results_train = tsvd.fit_transform(train_df)
tsvd_results_test = tsvd.transform(test_df)

# PCA
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(train_df)
pca2_results_test = pca.transform(test_df)

# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica2_results_train = ica.fit_transform(train_df)
ica2_results_test = ica.transform(test_df)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results_train = grp.fit_transform(train_df)
grp_results_test = grp.transform(test_df)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results_train = srp.fit_transform(train_df)
srp_results_test = srp.transform(test_df)

############
# write output
#############
print("Writing output...")
outputTrain = pd.DataFrame()
outputTest = pd.DataFrame()

outputTrain['id'] = train_id
outputTrain['target'] = train_target
outputTest['id'] = test_id

for i in range(1, n_comp + 1):
    outputTrain['pca_' + str(i)] = pca2_results_train[:, i - 1]
    outputTest['pca_' + str(i)] = pca2_results_test[:, i - 1]

    outputTrain['ica_' + str(i)] = ica2_results_train[:, i - 1]
    outputTest['ica_' + str(i)] = ica2_results_test[:, i - 1]

    outputTrain['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    outputTest['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    outputTrain['grp_' + str(i)] = grp_results_train[:, i - 1]
    outputTest['grp_' + str(i)] = grp_results_test[:, i - 1]

    outputTrain['srp_' + str(i)] = srp_results_train[:, i - 1]
    outputTest['srp_' + str(i)] = srp_results_test[:, i - 1]


filename_write_train = os.path.join(path, "data-{}-train.csv".format(NAME))
filename_write_test = os.path.join(path, "data-{}-test.csv".format(NAME))
filename_write_train2 = os.path.join(path, "data-{}-train.pkl".format(NAME))
filename_write_test2 = os.path.join(path, "data-{}-test.pkl".format(NAME))

#jhkaggle.util.save_pandas(outputTrain,filename_write_train)
jhkaggle.util.save_pandas(outputTrain,filename_write_train2)
#jhkaggle.util.save_pandas(outputTest,filename_write_test)
jhkaggle.util.save_pandas(outputTest,filename_write_test2)

