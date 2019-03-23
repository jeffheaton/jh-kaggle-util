import config
import jhkaggle
import os
import time
import json
import jhkaggle.util
import jhkaggle.loader
from jhkaggle.perturb_importance import calculate_importance_perturb

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#MODEL = "lgb-0p56161_20190322-140833"
#MODEL = "extree-0p409116_20190322-141616"
#MODEL = "keras-0p452822_20190322-141239"
MODEL = "xgboost-0p576026_20190322-135327"

start_time = time.time()
model = jhkaggle.loader.load_model(MODEL,"model-fold1")

imp = calculate_importance_perturb(model)
jhkaggle.util.save_importance_report(MODEL,imp)

elapsed_time = time.time() - start_time
print("Elapsed time: {}".format(jhkaggle.util.hms_string(elapsed_time)))
