from keras.datasets import mnist
from autokeras import ImageClassifier
# 导入MNIST数据，并将其分配到训练集和测试集中
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train)
# x_train = x_train.reshape(x_train.shape + (1,))
# x_test = x_test.reshape(x_test.shape + (1,))
# clf = ImageClassifier(verbose=True)
# clf.fit(x_train, y_train, time_limit=12 * 60)
# clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
# y = clf.evaluate(x_test, y_test)
# print(y)