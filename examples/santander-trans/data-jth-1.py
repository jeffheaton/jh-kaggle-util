# Jeff Heaton's Kaggle Utilities
# Copyright 2019 by Jeff Heaton, Open Source, Released under the Apache License
# For more information: https://github.com/jeffheaton/jh-kaggle-util
# 
# The data- files are used to generate features from the raw data files provided by Kaggle.
# These features individual feature files are then joined together (by joiner.py) to become
# the feature vector that training actually occurs on. 
import numpy as np
import config
import jhkaggle
import jhkaggle.util
from jhkaggle.util import PassThru

# from nltk.tokenize.moses import MosesTokenizer

encoding = 'utf-8'
global_target = 0

# Columns simply passed as is
pass_columns = ['var_'+str(x) for x in range(0,200)]

columns = [PassThru(pass_columns)]

gen = jhkaggle.util.GenerateDataFile("jth-1", columns)
#gen.max_lines = 10000
gen.run()
gen.report()
