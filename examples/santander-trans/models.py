import config
import jhkaggle
import jhkaggle.train_xgboost
import jhkaggle.train_keras
import jhkaggle.train_sklearn
import jhkaggle.train_lightgbm
from jhkaggle.joiner import perform_join
from jhkaggle.ensemble_glm import ensemble
import time
import sklearn
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

# Modify the code in this function to build your own XGBoost trainers
# It will br executed only when you run this file directly, and not when
# you import this file from another Python script.s
def run_xgboost():
    COMMON = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'silent': 1,
        'n_jobs': -1
    }

    params = {'learning_rate': 0.02, 'seed': 4242, 'max_depth': 2,  'colsample_bytree': 0.3}


    params = {**params, **COMMON}
    print(params)

    start_time = time.time()
    train = jhkaggle.train_xgboost.TrainXGBoost("1",params=params,run_single_fold=False)
    train.early_stop = 50
    train.rounds = 10000
    train.run()


def run_keras():
  train = jhkaggle.train_keras.TrainKeras("1",False)
  train.zscore = False
  train.run()

  elapsed_time = time.time() - start_time
  print("Elapsed time: {}".format(jhkaggle.util.hms_string(elapsed_time)))

def run_sklearn():
  n_trees = 100
  n_folds = 3

  # https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/
  alg_list = [
      ['rforest',RandomForestClassifier(n_estimators=1000, n_jobs=-1, verbose=1, max_depth=3)],
      ['extree',ExtraTreesClassifier(n_estimators = 1000,max_depth=3,n_jobs=-1)],
      ['adaboost',AdaBoostClassifier(base_estimator=None, n_estimators=600, learning_rate=1.0)],
      ['knn', sklearn.neighbors.KNeighborsClassifier(n_neighbors=5,n_jobs=-1)]
  ]

  start_time = time.time()
  for name,alg in alg_list:
      train = jhkaggle.train_sklearn.TrainSKLearn("1",name,alg,False)
      train.run()
      train = None

def run_lgb():
  os.environ['KMP_DUPLICATE_LIB_OK']='True'
  params = {
    'bagging_freq': 5,          
    'bagging_fraction': 0.38,   'boost_from_average':'false',   
    'boost': 'gbdt',             'feature_fraction': 0.04,     'learning_rate': 0.0085,
    'max_depth': -1,             'metric':'auc',                'min_data_in_leaf': 80,     'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,            'num_threads': 8,              'tree_learner': 'serial',   'objective': 'binary',
    'reg_alpha': 0.1302650970728192, 'reg_lambda': 0.3603427518866501,'verbosity': 1
  }

 
  train = jhkaggle.train_lightgbm.TrainLightGBM("1",params=params,run_single_fold=False)
  train.early_stop = 50
  train.run()

  


if __name__ == "__main__":
  start_time = time.time()
  run_xgboost()
  #run_sklearn()
  #run_lgb()
  #run_keras()

  elapsed_time = time.time() - start_time
  print("Elapsed time: {}".format(jhkaggle.util.hms_string(elapsed_time)))