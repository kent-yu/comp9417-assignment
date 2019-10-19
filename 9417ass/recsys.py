import pandas as pd
import numpy as np
import random
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy import sparse


header = ['userId', 'movieId', 'rating', 'timestamp']
df=pd.read_csv('ml-100k/u.data',sep='\t', names=header)

train_data, test_data = train_test_split(df, test_size=0.2)

def trans2dict(df):
    u2i_dict = {}
    for index, row in df.iterrows():
        user = int(row['userId'])-1
        item = int(row['movieId'])-1
        rating = row['rating']
        if user not in u2i_dict:
            u2i_dict[user] = {}
        u2i_dict[user][item] = rating
    return u2i_dict

train_u2i_score = trans2dict(train_data)
test_u2i_score = trans2dict(test_data)
i2u_score = {}
user_ave_score = {}


for user in train_u2i_score:
    if not user in user_ave_score:
        user_ave_score[user] = 0.0
        cnt = 0
    for item in train_u2i_score[user]:
        cnt += 1
        user_ave_score[user] += train_u2i_score[user][item]
        if not item in i2u_score:
            i2u_score[item] = {}
        i2u_score[item][user] = train_u2i_score[user][item]
    user_ave_score[user] = user_ave_score[user]/cnt

#memory-based collaborative filtering
#calculate the user-user similarity
def cal_sim(i2u_score):
    sim = {}
    norm = {}
    for item in i2u_score:
        user_score = i2u_score[item]
        for user in user_score:
            if not user in norm:
                norm[user] = 0
            norm[user] += user_score[user] * user_score[user]
            if user not in sim:
                sim[user] = {}
            for user2 in user_score:
                if user2 == user:
                    continue
                if user2 not in sim[user]:
                    sim[user][user2] = 0
                sim[user][user2] += user_score[user] * user_score[user2]
    for u in sim:
        for v in sim[u]:
            sim[u][v] /= math.sqrt(norm[u] * norm[v])
    return sim

#calculate unknown score using usercf
def cal_score_usercf(similarity, train_u2i_score, test_u2i_score, i2u_score, K, user_ave_score):
    pred = {}
    for user in test_u2i_score:
        if not user in pred:
            pred[user] = {}
        for item in test_u2i_score[user]:
            norm = 0.0
            if not item in pred[user]:
                pred[user][item] = 0.0
            cnt = 0
            for v, sim in similarity[user]:
                if not item in i2u_score:
                    norm = 1
                    break
                if not v in i2u_score[item]:
                    continue
                pred[user][item] += sim * (train_u2i_score[v][item] - user_ave_score[v])
                norm += abs(sim)
                cnt += 1
                if cnt == K:
                    break
            pred[user][item] = pred[user][item] / norm
            if user not in user_ave_score:
                pred[user][item] += 0.0
            else:
                pred[user][item] += user_ave_score[user]
    return pred

similarity = cal_sim(i2u_score)
sorted_similarity = {k: list(sorted(v.items(), key=lambda x: x[1], reverse=True)) for k, v in similarity.items()}

# model-based collaborative filtering
# train svd model using Gradient Descent to get the user-feature-matrix and item-feature-matrix
def svd_train(df, train_u2i_score, vec_size, steps=10, alpha=0.02, lamda=0.3, decay_rate=0.95):
    user_cnt = df.userId.unique().shape[0]
    item_cnt = df.movieId.unique().shape[0]
    p = np.matrix(np.random.rand(user_cnt, vec_size), dtype=np.longfloat)
    q = np.matrix(np.random.rand(item_cnt, vec_size), dtype=np.longfloat)
    for step in range(steps):
        print("start step %d" % (step+1))
        for u in train_u2i_score:
            for i in train_u2i_score[u]:
                pui = float(np.dot(p[u,:], q[i,:].T))
                eui = train_u2i_score[u][i] - pui
                for k in range(vec_size):
                    p[u,k] += alpha*(q[i,k]*eui - lamda*p[u,k])
                    q[i,k] += alpha*(p[u,k]*eui - lamda*q[i,k])
        alpha *= decay_rate
    return p,q

#calculate unknown score using svd model
def cal_score_svd(test_u2i_score, p, q):
    pred = {}
    for u in test_u2i_score:
        if not u in pred:
            pred[u] = {}
        for i in test_u2i_score[u]:
            if not i in pred[u]:
                pred[u][i] = 0.0
            pred[u][i] = float(np.dot(p[u,:], q[i,:].T))
    return pred

p, q = svd_train(df, train_u2i_score, 64)

#eavl using rmse
def cf_eval(pred, test, method = "rmse"):
    error = 0.0
    cnt = 0
    for user in pred:
        for item in pred[user]:
            error += (pred[user][item] - test[user][item]) * (pred[user][item] - test[user][item])
            if (pred[user][item] - test[user][item]) == float("inf"):
                print (user,item)
            cnt += 1
    error = math.sqrt(error/cnt)
    return error

pred_usercf = cal_score_usercf(sorted_similarity, train_u2i_score, test_u2i_score, i2u_score, 60, user_ave_score)
pred_svd = cal_score_svd(test_u2i_score, p, q)
eval_usercf = cf_eval(pred_usercf, test_u2i_score)
eval_svd = cf_eval(pred_svd, test_u2i_score)
print ('usercf  Rmse: %s' % (eval_usercf))
print ('svd  Rmse: %s' % (eval_svd))
