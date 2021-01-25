import tensorflow as tf
import autokeras as ak
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
               df["进项因子得分"], df["销项因子得分"], df["违约"]):
    X.append(list(tup))
# Initialize the structured data classifier.
clf = ak.StructuredDataClassifier(
    overwrite=True,
    max_trials=3) # It tries 3 different models.
# Feed the structured data classifier with training data.
clf.fit(
    # The path to the train.csv file.
    train_file_path,
    # The name of the label column.
    'survived',
    epochs=50)
# Predict with the best model.
predicted_y = clf.predict(test_file_path)
# Evaluate the best model with testing data.
print(clf.evaluate(test_file_path, 'survived'))