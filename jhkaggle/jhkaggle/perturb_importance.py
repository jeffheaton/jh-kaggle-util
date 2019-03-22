# Jeff Heaton's Kaggle Utilities
# Copyright 2019 by Jeff Heaton, Open Source, Released under the Apache License
# For more information: https://github.com/jeffheaton/jh-kaggle-util
# 
# This is used to determine feature importance using the feature perturb algorithm
import jhkaggle
import jhkaggle.util
import jhkaggle.const
import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm import tqdm

def calculate_importance_perturb(model):
  fit_type = jhkaggle.jhkaggle_config['FIT_TYPE']

  x = jhkaggle.util.load_pandas("train-joined-{}.pkl".format(model.data_source))

  mask_test = np.array(x['fold'] == 1)
  x = x[mask_test]

  x.drop("id",axis=1,inplace=True)
  x.drop("fold",axis=1,inplace=True)
  y = x['target']
  x.drop("target",axis=1,inplace=True)
  columns = x.columns
  x = x.values

  errors = []

  for i in tqdm(range(x.shape[1])):

    hold = np.array(x[:, i])
    np.random.shuffle(x[:, i])
    
    pred = model.predict_model(model.model,x)
    if fit_type == jhkaggle.const.FIT_TYPE_REGRESSION:
        error = metrics.mean_squared_error(y, pred)
    else:
        error = metrics.log_loss(y, pred)
        
    errors.append(error)
    x[:, i] = hold
    
  max_error = np.max(errors)
  importance = [e/max_error for e in errors]

  data = {'name':columns,'error':errors,'importance':importance}
  result = pd.DataFrame(data, columns = ['name','error','importance'])
  result.sort_values(by=['importance'], ascending=[0], inplace=True)
  result.reset_index(inplace=True, drop=True)
  return result