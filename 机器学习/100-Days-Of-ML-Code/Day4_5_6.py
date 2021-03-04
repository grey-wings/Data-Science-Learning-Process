
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, \
    StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

"""Day4,5,6 Logistic回归"""
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, -1].values

sc_X = StandardScaler()
X = sc_X.fit_transform(X)

"""训练"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=13)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

"""预测"""
y_pred = classifier.predict(X_test)

"""模型评估"""
# 在混淆矩阵中，cm[0, 0]是真阴性（实际为0，判断为0），cm[1, 0]是假阴性（实际为1，判断为0），
# cm[0, 1]是假阳性（实际为0，判断为1），cm[1, 1]是真阳性（实际为1，判断为1）
cm = confusion_matrix(y_test, y_pred)
print(cm)

X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
# meshgrid生成网格点坐标矩阵
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
# ravel将数组扁平化（展成一维）；flatten也可以用，效果一样，而且flatten不改变原来的数组而ravel会改变。
# contourf和contour绘制等高线，但前者会填充整个区域。
# 第三个参数是对应的高度值(eg.前三个参数分别设为x,y,z,则z可认为是f(x,y))
# 这里第三个参数的意义是对整个平面中所有点作预测，将预测为0的区域和预测为1的区域分别图上不同的颜色，并和后面画上去的
# 数据点进行比对。
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):  # unique去除数组中的重复数字
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                color=ListedColormap(('red', 'green'))(i), label=j)
# plt.scatter参数详解：https://www.cnblogs.com/shuaishuaidefeizhu/p/11359826.html
# 没有查到关于label的说法，但label可能和图例有关。如果不写会出警告，但是做出来的图没什么区别。

plt.title(' LOGISTIC(Training set)')
plt.xlabel(' Age')
plt.ylabel(' Estimated Salary')
plt.legend()
plt.show()

X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                color=ListedColormap(('red', 'green'))(i), label=j)

plt.title(' LOGISTIC(Test set)')
plt.xlabel(' Age')
plt.ylabel(' Estimated Salary')
plt.legend()
plt.show()
