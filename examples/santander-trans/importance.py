import config
import jhkaggle
import os
import time
import json
import jhkaggle.util
import jhkaggle.loader
from jhkaggle.perturb_importance import calculate_importance_perturb

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#MODEL = "rforest-0p776201_20190322-200919"
#MODEL = "keras-0p845986_20190323-114844"
#MODEL = "rforest-0p776201_20190322-200919"
#MODEL = "lgb-0p900275_20190323-073709"
MODEL = "xgboost-0p896861_20190323-184111"
#MODEL = "knn-0p537225_20190323-070741" # VERY slow

start_time = time.time()
model = jhkaggle.loader.load_model(MODEL,"model-fold1")

imp = calculate_importance_perturb(model)
jhkaggle.util.save_importance_report(MODEL,imp)

elapsed_time = time.time() - start_time
print("Elapsed time: {}".format(jhkaggle.util.hms_string(elapsed_time)))
