import numpy as np
import pandas as pd
from sklearn import impute, model_selection
from sklearn import preprocessing

"""读入csv文件"""
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values  # .values将DataFrame转为ndarray;df包括索引名字
y = dataset.iloc[:, 3].values  # 这里X读取的是除掉最后一列的数据，y是最后一列

"""处理丢失数据
详见：
https://scikit-learn.org/stable/modules/generated/
sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer"""
imputer = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3])  # 用切片是因为x的第0列是字符串，不能使用mean.
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

"""解析分类数据
分类数据是指含有标签值而不是数字值的变量例如"Yes"或"No".
需要把分类数据解析成数字。
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncod
er.html#sklearn.preprocessing.LabelEncoder
"""
# 常见的几种二值化编码函数：OneHotEncoder, LabelEncoder , LabelBinarizer
# 参考文章：https://mikejun.blog.csdn.net/article/details/81118610?utm_medium=dist
# ribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.control&dis
# t_request_id=b63c7a24-87a3-45fa-a1a0-f6ff9f73f024&depth_1-utm_source=distribut
# e.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.control
#
# OneHotEncoder:独热编码，直观来说就是有多少个状态就有多少比特，而且只有一个比特为1，其他全为0的一种码制。
# 离散特征进行one-hot编码后，编码后的特征，其实每一维度的特征都可以看做是连续的特征。
# 就可以跟对连续型特征的归一化方法一样，对每一维特征进行归一化。比如归一化到[-1,1]或归一化到均值为0,方差为1。
# 独热化编码后，所有向量到原点距离相同，没有偏序性。适合没有顺序的特征。
# 缺点：当类别的数量很多时，特征空间会变得非常大。在这种情况下，一般可以用PCA来减少维度。
# sklearn 的新版本中，OneHotEncoder 的输入必须是 2-D array
# OneHotEncoder无法直接对字符串型的类别变量编码
#
# LabelEncoder() 将转换成连续的数值型变量。即是对不连续的数字或者文本进行编号
#
# 无论 LabelEncoder() 还是 LabelBinarizer()，他们在 sklearn 中的设计初衷，都是为了解决标签y的离散化，
# 而非输入X， 所以他们的输入被限定为 1-D array，这恰恰跟OneHotEncoder() 要求输入 2-D array 相左。
labelencoder_X = preprocessing.LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
"""这里使用LabelEncoder是原仓库的用法。这一部分有很多没看懂的地方，但此时X为一维数组，
使用这种方法并没有什么问题。"""

"""
划分数据集
"""
X_train, X_test, y_train, y_test = model_selection.train_test_split\
    (X, y, test_size=0.2, random_state=4)
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
X_train, y_train, X_test, y_test = X_train.reshape(-1, 1), y_train.reshape(-1, 1), \
                                   X_test.reshape(-1, 1), y_test.reshape(-1, 1)
# 要训练的数据集必须是二维数组，要reshape一下

"""
特征量化
StandardScaler通过去除均值并缩放到单位方差来标准化特征。
"""
sc_X = preprocessing.StandardScaler()
X_train = sc_X.fit_transform(X_train)  # fit_transform:fit to data, then transform it.
X_test = sc_X.transform(X_test)  # transform:根据找到的规则对数据进行转换。
