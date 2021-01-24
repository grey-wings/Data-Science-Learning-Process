import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sklearn.metrics


# 读取数据
x, y = [], []
with open("ex1data1.txt", 'r') as f:
    while True:
        lines = f.readline()
        if not lines:
            break
        x1, y1 = [float(i) for i in lines.split(',')]
        x.append(x1)
        y.append(y1)
x = np.array(x)
y = np.array(y)
data = np.c_[x, y]     #数组按列（左右）拼接，按行（上下）拼接使用np.r_
print(data)

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=4)
# X_train, X_test, y_train, y_test =\
#     sklearn.model_selection.train_test_split\
#         (train_data, train_target, test_size=0.4, random_state=0, stratify=y_train)
# 该函数将数据集分为训练集和测试集
# train_data：所要划分的样本特征集
# train_target：所要划分的样本结果
# test_size：样本占比，如果是整数的话就是样本的数量
# random_state：是随机数的种子。
# 随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。
# 比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
x_train, y_train, x_test, y_test = x_train.reshape(-1, 1), y_train.reshape(-1, 1), \
                                   x_test.reshape(-1, 1), y_test.reshape(-1, 1)
# 要训练的数据集必须是二维数组，要reshape一下

# 画出图像
plt.scatter(data[:, 0], data[:, 1], marker='o')
plt.savefig("初始图.png")
# plt.show()
'''这句注释掉，否则后面画直线会吞掉点'''

# 回归
model = linear_model.LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
k = float(model.coef_)
b = float(model.intercept_)
# model.coef_和intercept_生成ndarray，必须将其转为float才能当参数


# 作图
x1 = np.linspace(4, 24, 200)  # 下界4，上界24，画200个点
y1 = k * x1 + b  # 直接写出函数解析式,也可以画二次函数等
plt.plot(x1, y1, '-r')  # 红色实线
plt.savefig("result.png")
plt.show()


# 均方根误差
# 均方误差是指参数估计值与参数真值之差平方的期望值，
# 记为MSE。MSE是衡量平均误差的一种较方便的方法，MSE可以评价数据的变化程度，
# MSE的值越小，说明预测模型描述实验数据具有更好的精确度。
print("均方根误差：", sklearn.metrics.mean_squared_error(y_test, y_pred))


