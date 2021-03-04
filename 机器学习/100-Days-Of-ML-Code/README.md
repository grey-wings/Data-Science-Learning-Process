# 目录
## [Day1 数据预处理](https://github.com/grey-wings/Data-Science-Learning-Process/blob/main/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/100-Days-Of-ML-Code/Day1.py)
### 1.导入库
### 2.读取csv文件
<font color=#FF0000> 注意！！！DataFrame.values是属性而不是方法，不能加括号，否则会报错：TypeError: 'numpy.ndarray' object is not callable </font>   
### 3.处理丢失数据
将nan等非数据值替换为平均值或中间值等。
### 4.解析分类数据
将标签值，如“yes”和“no”替换成能够用于计算的数字。
### 5.将数据拆分成训练集和测试集
许多sklearn函数要求输入数据集是二维数组，这里应根据具体要求进行reshape.
### 6.特征缩放
标准化数据，将其缩放到方差为0，均值为1.

## [Day2 简单线性回归（没写）](https://github.com/grey-wings/Data-Science-Learning-Process/blob/main/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/100-Days-Of-ML-Code/Day2.py)
### 1.数据预处理
所有步骤参照Day1
### 2.训练集使用简单线性回归模型来训练

## [Day3 多元线性回归](https://github.com/grey-wings/Data-Science-Learning-Process/blob/main/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/100-Days-Of-ML-Code/Day3.py)
### 1.数据预处理
所有步骤参照Day1
### 2.避免虚拟变量陷阱
### 3.输出参数和截距
### 4.预测
regressor.predict(X_test)

## [Day4,5,6 Logistic回归](https://github.com/grey-wings/Data-Science-Learning-Process/blob/main/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/100-Days-Of-ML-Code/Day4_5_6.py)
### 1.数据预处理
### 2.训练和预测
### 3.输出混淆矩阵
### 4.作图分析结果
np.meshgrid; plt.contourf; plt.scatter; plt作图和图例相关操作

## [Day11 K-NN（没写）]

## [Day9,10,11,12,14 SVM](https://github.com/grey-wings/Data-Science-Learning-Process/blob/main/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/100-Days-Of-ML-Code/Day9_10_11_12_14.py)
