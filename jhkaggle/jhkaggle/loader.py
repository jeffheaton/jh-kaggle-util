import jhkaggle.util
import os
import json
from jhkaggle.train_lightgbm import TrainLightGBM
from jhkaggle.train_sklearn import TrainSKLearn
from jhkaggle.train_keras import TrainKeras
from jhkaggle.train_xgboost import TrainXGBoost

def load_model(folder,name):
  # First, load the JSON meta
  root = jhkaggle.jhkaggle_config['PATH']
  model_path = os.path.join(root,folder)
  meta_filename = os.path.join(model_path,"meta.json")

  with open(meta_filename, 'r') as fp:
    meta = json.load(fp)

  model_name = meta['name']
  print(model_name)

  if model_name == 'TrainLightGBM':
    return TrainLightGBM.load_model(folder,name)
  elif model_name == 'TrainSKLearn':
    return TrainSKLearn.load_model(folder,name)
  elif model_name == 'TrainKeras':
    return TrainKeras.load_model(folder,name)
  elif model_name == 'TrainXGBoost':
    return TrainXGBoost.load_model(folder,name)
  else:
    raise Exception(f"Unknown model type: {model_name}")