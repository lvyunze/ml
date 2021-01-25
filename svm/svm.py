#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : yunze
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split   # 数据拆分模块
from sklearn.neighbors import KNeighborsClassifier
import sklearn.preprocessing, sklearn.decomposition, sklearn.linear_model, sklearn.pipeline, sklearn.metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
excel_data = pd.read_excel("./描述性统计分析.xls")
header_list = ["进项金额的均值", "进项金额的方差", "进项税额的均值", "进项税额的方差",
               "销项金额的均值", "销项金额的方差", "销项税额的均值", "销项税额的方差",
               "利润", "进项因子得分", "销项因子得分", "违约"]
df = pd.DataFrame(
    {data_item: excel_data[data_item] for data_item in header_list}
)
X = []
for tup in zip(df["进项金额的均值"], df["进项金额的方差"], df["进项税额的均值"],
               df["进项税额的方差"], df["销项金额的均值"], df["销项金额的方差"],
               df["销项税额的均值"], df["销项税额的方差"], df["利润"],
               df["进项因子得分"], df["销项因子得分"]):
    X.append(list(tup))
print(X)

train_X, test_X, train_y, test_y = train_test_split(X, list(df["违约"]), test_size=0.2, random_state=5)
print(train_X)
# print(data)
# test_random = np.random.randint(0, 122, 25)
# test_data = data.loc[test_random]
# train_random = [i for i in range(123) if i not in test_random]
# train_data = data.loc[train_random]
knn = KNeighborsClassifier()
# print(train_data["违约"])
# print(train_data[header_list[:-1]])
grid_param = {'n_neighbors': list(range(2, 10)),
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
clf = RandomizedSearchCV(knn, grid_param, n_iter=50)

clf.fit(train_X, train_y)
# #best parameter combination
print(clf.best_params_)


# score achieved with best parameter combination
print(clf.best_score_)

# all combinations of hyperparameters
print(clf.cv_results_['params'])

# average scores of cross-validation
print(clf.cv_results_['mean_test_score'])
