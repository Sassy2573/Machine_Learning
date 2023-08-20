import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

data = 0
class Perceptron():
    def __init__(self):
        # w初始化为全1数组
        self.w = np.ones(len(data[0]) - 1, dtype=np.float32)
        self.b = 0
        self.rate = 0.5  # 初始化学习率

    # 感知机训练, 找出最合适的w, b
    def fit(self, x_train, y_train):
        while True:
            flag = True  # 标记是否存在误分类数据
            for i in range(len(x_train)):  # 遍历训练数据
                xi = x_train[i]
                yi = y_train[i]
                # 判断 yi * (wx + b) <= 0
                if yi * (np.inner(self.w, xi) + self.b) <= 0:
                    flag = False  # 找到误分类数据, flag标记为False
                    # 更新w, b值
                    self.w += self.rate * np.dot(xi, yi)
                    self.b += self.rate * yi
            if flag:
                break
        # 输出w = ? , b = ?
        print('w = ' + str(self.w) + ', b = ' + str(self.b))

    # 图形显示结果
    def show(self, data):
        x_ = np.linspace(4, 7, 10)
        y_ = -(self.w[0] * x_ + self.b) / self.w[1]
        # 画出这条直线
        plt.plot(x_, y_)
        # 画出数据集的散点图
        plt.plot(data[:50, 0], data[:50, 1], 'bo', c='blue', label='0')
        plt.plot(data[50:100, 0], data[50:100, 1], 'bo', c='orange', label='1')
        plt.xlabel('sepal length')
        plt.ylabel('sepal width')
        plt.legend()
        plt.show()


iris = load_iris()
# 通过DataFrmae对象获取iris数据集对象, 列名为该数据集样本的特征名
df = pd.DataFrame(iris.data, columns = iris.feature_names)
# 增加label列为它们的分类标签
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
#print(df.label.value_counts()) 不同标签的样本数量
# 2:50 1:50 0:50


plt.scatter(df[:50]['sepal length'].values, df[:50]['sepal width'].values, label='0')
plt.scatter(df[50:100]['sepal length'].values, df[50:100]['sepal width'].values, label='1')
plt.scatter(df[100:150]['sepal length'].values, df[100:150]['sepal width'].values, label='2')

plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

# 选择数据集
data = np.array(df.iloc[:100, [0, 1, -1]])
# 数据集划分
x, y = data[:, :-1], data[:, -1]
# 将y数据集值变为1和-1
y = np.array([1 if i == 1 else -1 for i in y])

# 开始训练
p = Perceptron()
p.fit(x, y)
p.show(data)