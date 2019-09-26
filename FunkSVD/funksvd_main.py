import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import math

'''
FunkSVD,是一种伪SVD过程，也称为LFM, 实验使用ML-1M数据集

'''
filepath = '../datasets/ml-1m/ratings.dat'
header = ['UserID','MovieID','Rating','Timestamp']
data = pd.read_csv(data,sep='::',names=header)

#用户ID或者MoiveID不连续，需要转化成连续的ID
def map2idx(data,col_name):
    unique_list = data[col_name].unique().tolist()
    src2id_map = dict(zip(unique_list,range(len(unique_list))))
    id2src_map = dict(zip(range(len(unique_list)) ,unique_list))
    data[col_name] = data[col_name].map(src2id_map)
    return data, src2id_map, id2src_map



rateing_df,mov_src2id_map,mov_id2src_map = map2idx(rateing_df,'MovieID')
rateing_df,usr_src2id_map,usr_id2src_map = map2idx(rateing_df,'UserID')

num_users = rateing_df['UserID'].nunique() #统计有多少不同的用户
num_items = rateing_df['MovieID'].nunique() #统计有多少不同的物品

#划分训练集和测试集合
train_data ,test_data = train_test_split(rateing_df,test_size=0.15)

#初始化用户向量和物品向量
def initModel(train,k):

    P = {}
    Q = {}
    for _,row in train.iterrows():
        P[row['UserID']] = 0.1 * np.random.random(k)
        Q[row['MovieID']] = 0.1 * np.random.random(k)
    return P,Q

#进行样本的负采样
def RandomSelectNegativeSample(user_items,all_items,ratio=1):

    dict = {}
    for item in user_items:
        dict[item] = 1
    n_sample = min(int(len(user_items) * ratio),len(all_items))
    for i in range(n_sample):
        item = all_items[random.randint(0,len(all_items) - 1)]
        while item in user_items:
            item = all_items[random.randint(0, len(all_items) - 1)]
        dict[item] = -1
    return dict

#通过算处理的矩阵，进行评分预测
def Predict(user_id,item_id,P,Q):
    user_vec = P[user_id]
    item_vec = Q[item_id]
    rate = np.dot(user_vec,item_vec)
    return rate


#向用户推荐k个物品
def Recommandation(user_id,k,P,Q,item_user,all_items):

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

    hit = 0
    all = 0

    user_list = test['UserID'].unique().tolist()
    all_items = train['MovieID'].unique().tolist()

    for user in user_list:
        train_item = train[train['UserID'] == user].MovieID.unique().tolist()
        test_item = test[test['UserID'] == user].MovieID.unique().tolist()
        recommand_list = Recommandation(user,N,P,Q,train_item,all_items)
        #print(train_moive,test_moive,recommand_list)
        #print(recommand_list)

        all += len(test_item)
        hit += len(list(set(test_item).intersection(set(recommand_list))))

    return hit / (all * 1.0)

#准确率计算
def Precision(train, test, N, P,Q):
    hit = 0
    all = 0

    user_list = test['UserID'].unique().tolist()
    all_items = train['MovieID'].unique().tolist()

    for user in user_list:
        train_item = train[train['UserID'] == user].MovieID.unique().tolist()
        test_item = test[test['UserID'] == user].MovieID.unique().tolist()
        recommand_list = Recommandation(user, N, P, Q, train_item,all_items)
        #print(recommand_list)
        all += N
        hit += len(list(set(test_item).intersection(set(recommand_list))))

    return hit / (all * 1.0)

#SVD的训练过程
def FunkSVD_train(train,k,step,alpha,lambdas):
    P,Q = initModel(train,k)

    all_items = train['MovieID'].tolist()
    user_list = train['UserID'].unique().tolist()

    print(len(user_list))
    print(len(all_items))

    for s in range(step):
        print('step:',s)
        i = 0

        for user in user_list:
            if i % 200 == 0:
                print('user',i)
            i = i + 1
            user_items = train[train['UserID'] == user].MovieID.tolist()
            user_samples = RandomSelectNegativeSample(user_items,all_items,2)

            for item, label in user_samples.items():
                rate = Predict(user,item,P,Q)
                loss_grad = label - rate
                #print('rate',rate)

                if math.fabs(rate) > 100:
                    print(P[user],Q[item])
                    print(rate)

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





P,Q = FunkSVD_train(train_data,50, 100, 0.02,0.01)

print('Recall',Recall(train_data,test_data,50,P,Q))
print('Precision',Precision(train_data,test_data,50,P,Q))















