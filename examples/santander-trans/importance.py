import config
import jhkaggle
import os
import time
import jhkaggle.util
from jhkaggle.train_lightgbm import TrainLightGBM
from jhkaggle.train_sklearn import TrainSKLearn
from jhkaggle.perturb_importance import calculate_importance_perturb

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#MODEL = "lgb-0p900275_20190321-071244"
MODEL = "extree-0p851132_20190321-163210"

start_time = time.time()
#model = TrainLightGBM.load_model(MODEL,"model-fold1")
model = TrainSKLearn.load_model(MODEL,"model-fold1")
imp = calculate_importance_perturb(model)
jhkaggle.util.save_importance_report(MODEL,imp)

elapsed_time = time.time() - start_time
print("Elapsed time: {}".format(jhkaggle.util.hms_string(elapsed_time)))
