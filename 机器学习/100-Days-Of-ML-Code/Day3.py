
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,\
    StandardScaler
from sklearn.linear_model import LinearRegression

"""Day3 多元线性回归"""
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(X)

"""虚拟变量陷阱
eg.有A、B、C三个类别，三个特征变量分别表示是否属于A、是否属于B、是否属于C
但是仅需前两个值就可以推出第三个。"""

labelencoder = LabelEncoder()
X[:, -1] = labelencoder.fit_transform(X[:, -1])
onehotencoder = OneHotEncoder()
q = onehotencoder.fit_transform(X[:, -1].reshape(-1, 1)).toarray()
q = q[:, 1:]
X = np.delete(X, -1, axis=1)
# axis=1代表删除列，axis=0代表删除行
# 删除多列：np.delete(arr, [1,2], axis=1)
X = np.append(X, q, axis=1)
# 当arr的维数为2(理解为单通道图)，axis=0表示沿着行方向添加values；axis=1表示沿着列方向添加values
# 当arr的维数为3(理解为多通道图)，axis=0，axis=1时同上；axis=2表示沿着深度方向添加values

sc_X = StandardScaler()
X = sc_X.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
regressor = LinearRegression().fit(X_train, y_train)
print(regressor.coef_)
print(regressor.intercept_)

