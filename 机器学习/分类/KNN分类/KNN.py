import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


# KNN基本知识
# 对未知类别属性的数据集中的每个点依次执行以下操作：
# 1.计算已知类别数据集中的点与当前点之间的距离；
# 2.按照距离递增次序排序；
# 3.选取与当前点距离最小的k个点；
# 4.确定前k个点所在类别的出现频率；
# 5.返回前k个点出现频率最高的类别作为当前点的预测分类。
#
# 使用函数：
# sklearn.neighbors.KNeighborsClassifier(n_neighbors=5,
# weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’,
# metric_params=None, n_jobs=1, **kwargs)
# 调用时令n_neighbors=k,k需要通过尝试去得出最好的取值。


# 读取文件
def txt_to_ndarray(FilePath):
    '''
    本函数读取txt文件（KNN的例程），并返回数据和标签。
    可根据需要进行修改。
    :param FilePath: 要读取的txt文件路径
    :return datamat: 数据集（ndarray）
            labelmat: 标签集（list）
    '''
    with open(FilePath, 'r') as f:
        lines = f.readlines()
        rows = len(lines)  # 文件行数
        lin = lines[0]
        s = lin.split('\t')
        datamat = np.zeros((rows, len(s)-1))  # 初始化矩阵
        labelmat = []
        row = 0
        for line in lines:
            line = line.strip().split('\t')  # strip()默认移除字符串首尾空格或换行符
            datamat[row, :] = line[0:len(s)-1]
            labelmat.append(line[-1])
            row += 1
    return datamat, labelmat


# 由于三个特征单位不同，且数字插值最大的属性对计算结果的影响最大，因此需要对其进行数据归一化的处理
# 归一化是将样本的特征值转换到同一量纲下把数据映射到[0,1]或者[-1, 1]区间内，
# 仅由变量的极值决定，因区间放缩法是归一化的一种。
def normalization(dataSet):
    '''
    本函数采用min-max归一化，
    :param dataSet: 要归一化的ndarray
    :return normDataSet:归一化完成的ndarray
    '''
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))  # tile函数的介绍建议help
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet


X, y = txt_to_ndarray(r'D:\SME\SME-数据\书籍\机器学习\机器学习实战 数据集\Ch02\datingTestSet.txt')
X = normalization(X)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=1/3,random_state=3)
# 这里划分数据以1/3的来划分 训练集训练结果 测试集测试结果
k_range = range(1, 45)
cv_scores = []  # 用来放每个模型的结果值
for n in k_range:
    knn = KNeighborsClassifier(n)
    # knn模型，这里一个超参数可以做预测，当多个超参数时需要使用另一种方法GridSearchCV
    scores = cross_val_score(knn, train_X, train_y, cv=10, scoring='accuracy')
    # 交叉验证
    # 本函数参考：https: // blog.csdn.net / qq_36523839 / article / details / 80707678
    # cv：选择每次测试折数  accuracy：评价指标是准确度,可以省略使用默认值，具体使用参考下面。
    cv_scores.append(scores.mean())

plt.plot(k_range, cv_scores, 'r-')
plt.xlabel('K')
plt.ylabel('Accuracy')  # 通过图像选择最好的参数
plt.show()

k_best = cv_scores.index(max(cv_scores))
print("精确度：", cv_scores[k_best], sep='')

best_knn = KNeighborsClassifier(n_neighbors=k_best)	# 选择最优的k传入模型
best_knn.fit(train_X, train_y)			            #训练模型
print("评分：", best_knn.score(test_X, test_y)*100, sep='')	#看看评分