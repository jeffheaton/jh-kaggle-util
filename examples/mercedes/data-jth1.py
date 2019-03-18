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


pass_columns = ['X'+str(x) for x in range(10,386)]
pass_columns.remove('X25') # Not defined
pass_columns.remove('X72') # Not defined
pass_columns.remove('X121') # Not defined
pass_columns.remove('X149') # Not defined
pass_columns.remove('X188') # Not defined
pass_columns.remove('X193') # Not defined
pass_columns.remove('X303') # Not defined
pass_columns.remove('X381') # Not defined

# X17: ['X382']
pass_columns.remove('X382')
# X31: ['X35', 'X37']
pass_columns.remove('X35')
pass_columns.remove('X37')
# X33: ['X39']
pass_columns.remove('X39')
# X44: ['X302']
pass_columns.remove('X302')
# X48: ['X113', 'X134', 'X147', 'X222']
pass_columns.remove('X113')
pass_columns.remove('X134')
pass_columns.remove('X147')
pass_columns.remove('X222')
# X53: ['X102', 'X214', 'X239']
pass_columns.remove('X102')
pass_columns.remove('X214')
pass_columns.remove('X239')
# X54: ['X76']
pass_columns.remove('X76')
# X58: ['X324']
pass_columns.remove('X324')
# X60: ['X253', 'X385']
pass_columns.remove('X253')
pass_columns.remove('X385')
# X62: ['X172', 'X216']
pass_columns.remove('X172')
pass_columns.remove('X216')
# X67: ['X213']
pass_columns.remove('X213')
# X71: ['X84', 'X244']
pass_columns.remove('X84')
pass_columns.remove('X244')
# X112: ['X199']
pass_columns.remove('X199')
# X118: ['X119']
pass_columns.remove('X119')
# X125: ['X227']
pass_columns.remove('X227')
# X138: ['X146']
pass_columns.remove('X146')
# X152: ['X226', 'X326']
pass_columns.remove('X226')
pass_columns.remove('X326')
# X155: ['X360']
pass_columns.remove('X360')
# X184: ['X262']
pass_columns.remove('X262')
# X230: ['X254']
pass_columns.remove('X254')
# X232: ['X279']
pass_columns.remove('X279')
# X240: ['X364']
pass_columns.remove('X364')
# X290: ['X293', 'X330']
pass_columns.remove('X293')
pass_columns.remove('X330')
# X295: ['X296']
pass_columns.remove('X296')
# X298: ['X299']
pass_columns.remove('X299')

class BooleanTFIDF:
    def __init__(self, name, columns):
        self.columns = columns
        self.name = ["btfidf-max","btfidf-min","btfidf-mean"]
        self.col_count = {}
        self.rows = 0

    def preprocess(self, header_idx, row):
        self.rows += 1
        for col in self.columns:
            value = int(row[header_idx[col]])
            if value == 1:
                self.col_count[col] = self.col_count.get(col,0) + 1

    def begin(self):
        for key in self.col_count.keys():
            self.col_count[key] = self.col_count[key] / float(self.rows)

    def process(self, header_idx, row):
        max_value = None
        min_value = None
        mean_value = 0
        cnt = 0
        for col in self.columns:
            value = int(row[header_idx[col]])
            if value == 0:
                value = self.col_count[col]
                mean_value += value
                cnt += 1

                if max_value == None or value>max_value:
                    max_value = value

                if min_value == None or value<min_value:
                    min_value = value

        mean_value /= float(cnt)
        return [max_value,min_value,mean_value]

columns = [
    BooleanTFIDF("btfidf",pass_columns),
    jhkaggle.util.SumColumns('bool_count',pass_columns),
    #CatMeanTarget('y','X0'),
    #CatMeanTarget('y','X1'),
    #CatMeanTarget('y','X2'),
    #CatMeanTarget('y','X3'),
    #CatMeanTarget('y','X4'),
    #CatMeanTarget('y','X5'),
    #CatMeanTarget('y','X6'),
    #CatMeanTarget('y','X8'),
    #OneHot('X0'),

    ##OneHot('X1'),
    ##OneHot('X2'),
    ##OneHot('X3'),
    ##OneHot('X4'),
    #OneHot('X5'),
    ##OneHot('X6'),
    ##OneHot('X8'),

    #PassThru(pass_columns)
]


gen = jhkaggle.util.GenerateDataFile("jth-1", columns)
#gen.max_lines = 10000
gen.run()
gen.report()
