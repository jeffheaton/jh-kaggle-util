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
from sklearn.linear_model import LinearRegression

# Modify the code in this function to build your own XGBoost trainers
# It will br executed only when you run this file directly, and not when
# you import this file from another Python script.s
def run_xgboost():
    COMMON = {
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }

    # params = {'scale_pos_weight': 1, 'max_depth': 2, 'subsample': 0.9, 'seed': 42, 'gamma': 0.0, 'colsample_bytree': 0.9, 'learning_rate': 0.005, 'reg_alpha': 1, 'silent': 1, 'min_child_weight': 9}
    #params = {'colsample_bytree': 0.9, 'gamma': 0.0, 'learning_rate': 0.01, 'min_child_weight': 9, 'reg_alpha': 0.01, 'seed': 42, 'subsample': 0.9, 'scale_pos_weight': 1, 'max_depth': 2}
    #params = {'colsample_bytree': 0.6, 'min_child_weight': 9, 'subsample': 0.6, 'max_depth': 2, 'reg_alpha': 0.1, 'seed': 42, 'learning_rate': 0.005, 'gamma': 0.0}
    #params = {'learning_rate' : 0.01, 'colsample_bytree': 0.2, 'subsample' : 1.0, 'max_depth' : 7, 'min_child_weight': 10, 'seed' : 4242}

    #params = {'min_child_weight': 7, 'reg_alpha': 0.01, 'gamma': 0.0, 'max_depth': 3, 'subsample': 0.7, 'scale_pos_weight': 1, 'learning_rate': 0.01, 'seed': 42, 'colsample_bytree': 0.6}
    #params = {'base_score': 100.669318128,'reg_alpha': 0.1, 'colsample_bytree': 0.6, 'learning_rate': 0.005, 'gamma': 0.0, 'seed': 42, 'min_child_weight': 9, 'max_depth': 2, 'subsample': 0.6, 'eval_metric': 'rmse'}
    params = {'base_score': 100.669318128, 'learning_rate': 0.005, 'scale_pos_weight': 1, 'colsample_bytree': 0.7, 'min_child_weight': 9, 'subsample': 0.6, 'max_depth': 2, 'silent': 1, 'gamma': 0.0, 'seed': 42, 'reg_alpha': 0.01}
    params = {'scale_pos_weight': 1, 'seed': 42, 'learning_rate': 0.005, 'base_score': 100.669318128, 'colsample_bytree': 0.6, 'max_depth': 2, 'gamma': 0.0, 'reg_alpha': 1, 'silent': 1, 'subsample': 0.7, 'min_child_weight': 9}
    params = {'learning_rate':0.0045,'base_score': 100.669318128,'seed':4242}


    params = {'max_depth': 2, 'subsample': 0.9, 'reg_alpha': 100, 'gamma': 0.0, 'min_child_weight': 7, 'seed': 4242, 'colsample_bytree': 0.9, 'silent': 1, 'base_score': 100.669318128, 'learning_rate': 0.0045}
    params = {'reg_alpha': 1e-05}
    params = {'silent': 1, 'learning_rate': 0.0045, 'seed': 4242, 'max_depth': 3, 'min_child_weight': 1, 'gamma': 0.0, 'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_alpha': 1e-05}


    params = {**params, **COMMON}
    print(params)

    start_time = time.time()
    train = jhkaggle.train_xgboost.TrainXGBoost("1",params=params,run_single_fold=False)
    train.early_stop = 50
    train.rounds = 10000
    #train.run_cv()
    train.run()

    elapsed_time = time.time() - start_time
    print("Elapsed time: {}".format(jhkaggle.util.hms_string(elapsed_time)))

def run_keras():
  # "all the time" to "always"
  # reall short ones that are dead wrong
  # [100]	train-logloss:0.288795	eval-logloss:0.329036
  # [598]	train-logloss:0.152968	eval-logloss:0.296854
  # [984]	train-logloss:0.096444	eval-logloss:0.293915

  start_time = time.time()
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
      ['lreg',LinearRegression()],
      ['rforest',RandomForestRegressor(n_estimators=1000, n_jobs=-1, max_depth=3)],
      ['extree',ExtraTreesClassifier(n_estimators = 1000,max_depth=2)],
      ['adaboost',AdaBoostRegressor(base_estimator=None, n_estimators=600, learning_rate=1.0)],
      ['knn', sklearn.neighbors.KNeighborsRegressor(n_neighbors=5)]
  ]

  start_time = time.time()
  for name,alg in alg_list:
      train = jhkaggle.train_sklearn.TrainSKLearn("1",name,alg,False)
      train.run()
      train = None
  elapsed_time = time.time() - start_time
  print("Elapsed time: {}".format(jhkaggle.util.hms_string(elapsed_time)))

def run_lgb():
  os.environ['KMP_DUPLICATE_LIB_OK']='True'
  params = {
    'metric':'rmse', 'num_threads': -1, 'objective': 'regression', 'verbosity': 1
  }
  train = jhkaggle.train_lightgbm.TrainLightGBM("1",params=params,run_single_fold=False)
  train.early_stop = 50
  train.run()

def run_ensemble():
  MODELS = [
    'xgboost-0p576026_20190319-181720',
    'keras-0p463293_20190319-185422'
  ]
  ensemble(MODELS)


if __name__ == "__main__":
    #run_lgb()
    run_xgboost()
    #run_keras()
    #run_sklearn()
    #run_ensemble()

