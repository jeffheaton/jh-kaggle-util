import config
import jhkaggle
from jhkaggle.ensemble_glm import ensemble
import time
import sklearn

def run_ensemble():
  MODELS = [
    'xgboost-0p89699_20190318-085921',
    'extree-0p856016_20190318-071430',
    'rforest-0p777229_20190318-065918',
    'knn-0p537225_20190318-025618',
    'lgb-0p900275_20190317-174005'
  ]
  ensemble(MODELS)


if __name__ == "__main__":
    run_ensemble()

