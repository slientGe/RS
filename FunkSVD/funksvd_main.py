import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import pickle
import math

'''
FunkSVD,是一种伪SVD过程，也称为LFM, 实验使用ML-1M数据集

'''
filepath = '../datasets/ml-1m/ratings.dat'
header = ['UserID','MovieID','Rating','Timestamp']
data = pd.read_csv(filepath,sep='::',names=header)


def map2idx(data,col_name):
    '''
    将不连续的特征进行重新编号
    :param data: 输入的数据 pd.DataFrame
    :param col_name: 需要处理的列
    :return: 返回重新编号后的数据及映射表
    '''
    unique_list = data[col_name].unique().tolist()
    src2id_map = dict(zip(unique_list,range(len(unique_list))))
    id2src_map = dict(zip(range(len(unique_list)) ,unique_list))
    data[col_name] = data[col_name].map(src2id_map)
    return data, src2id_map, id2src_map





#初始化用户向量和物品向量
def _initModel(train,k):
    '''
    初始化user和item对应的隐向量，生成P和Q矩阵
    :param train:
    :param k:
    :return:
    '''

    P = {}
    Q = {}
    for _,row in train.iterrows():
        P[row['UserID']] = 0.1 * np.random.random(k)
        Q[row['MovieID']] = 0.1 * np.random.random(k)
    return P,Q


def RandomSelectNegativeSample(user_items,all_items,ratio=1):
    '''
    进行负采样生成负样本，正样本标签为1，负样本标签为0
    :param user_items: user历时交互过的item列表
    :param all_items: 所有的item构成的列表
    :param ratio: 负样本与正样本的比例
    :return:
    '''

    dict = {}
    for item in user_items:
        dict[item] = 1
    n_sample = min(int(len(user_items) * ratio),len(all_items))
    for i in range(n_sample):
        item = all_items[random.randint(0,len(all_items) - 1)]
        while item in user_items:
            item = all_items[random.randint(0, len(all_items) - 1)]
        dict[item] = 0
    return dict


def Predict(user_id,item_id,P,Q):
    '''
        预测user_id对应的用户对item_id产生的评分
    '''
    user_vec = P[user_id]
    item_vec = Q[item_id]
    rate = np.dot(user_vec,item_vec)
    return rate



def Recommandation(user_id,k,P,Q,item_user,all_items):
    '''
        向指定user推荐k个商品
    '''
    user_vector = P[user_id]

    score_list  = []
    for item in all_items:
        item_vector = Q[item]
        score = np.dot(user_vector,item_vector)
        score_list.append(score)
    user_item_rate = list(zip(all_items,score_list))
    num = 0
    recommand_list = []
    for item in sorted(user_item_rate, key=lambda x:x[1],reverse=True):
        item_id = item[0]
        if item_id in item_user:
            continue
        recommand_list.append(item_id)
        num += 1
        if num > k:
            break
    return recommand_list


#召回率计算
def Recall(train,test,N,P,Q):
    '''
        Top N 的召回率计算
    '''
    hit = 0
    all = 0

    user_list = test['UserID'].unique().tolist()
    all_items = train['MovieID'].unique().tolist()

    for user in user_list:
        train_items = train[train['UserID'] == user].MovieID.unique().tolist()
        test_items = test[test['UserID'] == user].MovieID.unique().tolist()
        recommand_list = Recommandation(user,N,P,Q,train_items,all_items)
        all += len(test_items)
        hit += len(list(set(test_items).intersection(set(recommand_list))))

    return hit / (all * 1.0)


def Precision(train, test, N, P,Q):
    '''
        Top N 精确率计算
    '''
    hit = 0
    all = 0

    user_list = test['UserID'].unique().tolist()
    all_items = train['MovieID'].unique().tolist()

    for user in user_list:
        train_items = train[train['UserID'] == user].MovieID.unique().tolist()
        test_items = test[test['UserID'] == user].MovieID.unique().tolist()
        recommand_list = Recommandation(user, N, P, Q, train_items,all_items)
        all += N
        hit += len(list(set(test_items).intersection(set(recommand_list))))

    return hit / (all * 1.0)


def FunkSVD_train(train,k,step,alpha,lambdas,ratio = 2):
    '''

    :param train: 训练集
    :param k: 隐藏向量维度
    :param step: 训练轮数
    :param alpha: 学习率
    :param lambdas: 正则化参数
    :return:
    '''

    P,Q = _initModel(train,k)
    all_items = train['MovieID'].tolist()
    user_list = train['UserID'].unique().tolist()
    print(len(user_list))
    print(len(all_items))

    for s in range(step):
        print('step:',s)

        for user in user_list:
            user_items = train[train['UserID'] == user].MovieID.tolist()
            user_samples = RandomSelectNegativeSample(user_items,all_items,ratio)

            for item, label in user_samples.items():
                rate = Predict(user,item,P,Q)
                loss_grad = label - rate
                user_vector = P[user]
                item_vector = Q[item]
                #print(loss_grad)
                user_vector += alpha * (loss_grad * item_vector - lambdas * user_vector)
                item_vector += alpha * (loss_grad * user_vector - lambdas * item_vector)
                P[user] = user_vector
                Q[item] = item_vector
        if s % 2 == 0:
            alpha *= 0.9
            print('Recall', Recall(train_data, test_data, 50, P, Q))
            print('Precision', Precision(train_data, test_data, 50, P, Q))

    return P,Q


if __name__ == '__main__':
    filepath = '../datasets/ml-1m/ratings.dat'
    header = ['UserID', 'MovieID', 'Rating', 'Timestamp']

    data = pd.read_csv(filepath, sep='::', names=header)

    data, mov_src2id_map, mov_id2src_map = map2idx(data, 'MovieID')
    data, usr_src2id_map, usr_id2src_map = map2idx(data, 'UserID')


    train_data, test_data = train_test_split(data, test_size=0.15)

    P,Q = FunkSVD_train(train_data,50, 1, 0.02,0.01)

    print('Recall',Recall(train_data,test_data,50,P,Q))
    print('Precision',Precision(train_data,test_data,50,P,Q))

    with open('funksvd_result.pkl',mode='wb') as f:
        pickle.dump(P,f,pickle.HIGHEST_PROTOCOL)
        pickle.dump(Q,f,pickle.HIGHEST_PROTOCOL)
















