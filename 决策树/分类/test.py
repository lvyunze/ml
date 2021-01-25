# from sklearn.datasets import load_iris
# from sklearn import tree
# import pydotplus
# import os
# os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'  # 注意修改你的路径
# iris = load_iris()
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(iris.data, iris.target)
#
# dot_data = tree.export_graphviz(clf, out_file=None,
#                          feature_names=iris.feature_names,
#                          class_names=iris.target_names,
#                          filled=True, rounded=True,
#                          special_characters=True)
#
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_gif('iris.gif')

from queue import PriorityQueue
# 存储数据时可设置优先级的队列
# 优先级设置数越小等级越高
pq = PriorityQueue(maxsize=0)

# 写入队列，设置优先级,可以知道优先级分别为1、7、9
pq.put((9, 'a'))
pq.put((7, 'c'))
pq.put((1, 'd'))

# 输出队例全部数据
print(pq.queue)

# # 取队例数据，可以看到，是按优先级取的。
print(pq.get())
print(pq.get())
print(pq.get())
# pq.get()

