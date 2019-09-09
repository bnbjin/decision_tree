import numpy as np


def createDataSet():
    '''
    创建数据集

    returns:
        dataset:    训练决策树模型的数据集
        labels:     特征名/属性名
    '''

    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']

    return dataset, labels

def calcShannonEnt(dataSet):
    '''
    计算香农熵entropy
    '''

    # 计算标签种类及数量
    labelCounts = {}
    for featVec in dataSet:
        # 最后一个分量/特征 是 标签
        currentLabel = featVec[-1]

        # 若当前的标签还未统计，则添加对应键
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0

        # 统计对应标签的总数
        labelCounts[currentLabel] += 1

    # 计算香农熵
    shannonEnt = 0.0

    for key in labelCounts:
        # p(xi) = 标签数 / 总标签数（即数据集大小）
        prob = float(labelCounts[key]) / len(dataSet)
        shannonEnt -= prob * log(prob,2) #log base 2

    return shannonEnt

def splitDataSet(dataSet, axis, value):
    '''
    根据特征值分离数据集
    从数据集中满足特征value的输入变量剔除value的特征项
    '''

    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 指定特征列前部分
            reducedFeatVec = featVec[:axis]
            # 指定特征列后部分
            reducedFeatVec.extend(featVec[axis+1:])

            retDataSet.append(reducedFeatVec)

    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    '''
    '''

    # the last column is used for the labels
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1

    # iterate over all the features
    for i in range(numFeatures):
        # create a list of all the examples of this feature
        featList = [example[i] for example in dataSet]
        # get a set of unique values
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)

        # calculate the info gain; ie reduction in entropy
        infoGain = baseEntropy - newEntropy

        # compare this to the best gain so far
        if (infoGain > bestInfoGain):
            # if better than current best, set to best
            bestInfoGain = infoGain
            bestFeature = i

    # returns an integer
    return bestFeature
