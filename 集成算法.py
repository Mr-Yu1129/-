# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

#构建评分函数
from sklearn.metrics import make_scorer, mean_squared_error
# metric for evaluation
def rmse(y_true, y_pred):
    diff = y_pred - y_true
    sum_sq = sum(diff**2)    
    n = len(y_pred)       
    return np.sqrt(sum_sq/n)
def mse(y_ture,y_pred):
    return mean_squared_error(y_ture,y_pred)
# scorer to be used in sklearn model fitting
rmse_scorer = make_scorer(rmse, greater_is_better=False)
mse_scorer = make_scorer(mse, greater_is_better=False)

#from sklearn.model_selection import KFold
#from scipy import sparse
import lightgbm
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import GradientBoostingRegressor


#这是分步骤
#输入是：分类器对象，训练集数据，训练集标签，测试集数据，分类器的名称，交叉验证的折数，
#返回是：train和test。
#train是交叉验证过程中对训练集所有数据的预测，因为交叉验证都要留1/k用于验证，所以验证的部分会遍历整个训练集，
#于是就可以得到整个训练集的预测。test是在交叉验证每一折上模型对test_x的预测结果的平均，就是一个稳定版的test_x预测结果。
#作用是：得到该分类器在整个训练集上的预测结果与在测试集上稳定的结果（因为取了5次（因为是5折）预测的平均）

def stacking_reg(clf, train_x, train_y, test_x, clf_name, kf,
                 label_split=None):  
#下面四个都是先弄好存储的地方    
    train = np.zeros((train_x.shape[0], 1))# 整个地方存储交叉验证预测结果（先初始化）
    test = np.zeros((test_x.shape[0], 1))# 模型在test_x上的预测结果
    test_pre = np.empty((folds, test_x.shape[0], 1))# 创建对应维度的数组，保存k折交叉验证在test_x上的预测结果
    cv_scores = []# 用来保存交叉验证上每一折测试的mse 
    

    #用指定分类器来进行k折交叉验证
    for i, (train_index,test_index) in enumerate(kf.split(train_x, label_split)): #enumerate后面融合的时候用到
        # kf.split 返回的是训练集测试集的index，通过index获得训练集和测试集
        tr_x = train_x[train_index]
        tr_y = train_y[train_index]
        te_x = train_x[test_index]
        te_y = train_y[test_index]
        # 如果模型在该list中，fit与predict的方法都是一样的
        if clf_name in ["rf", "ada", "gb", "et", "lr", "lsvc", "knn"]:
            # 训练与预测
            clf.fit(tr_x, tr_y)
            pre = clf.predict(te_x).reshape(-1, 1)           
            train[test_index] = pre # 将这一折的预测结果放入对应位置。            
            test_pre[i, :] = clf.predict(test_x).reshape(-1, 1) # 预测test_x的数据（上面的te_x是k折中用于测试的部分，而test_x是单独的测试的部分）        
            cv_scores.append(mean_squared_error(te_y, pre)) # 计算该折下模型的mse 

        # lgb模型需要另外定义训练方式
        elif clf_name in ["lgb"]:
            train_matrix = clf.Dataset(tr_x, label=tr_y)
            test_matrix = clf.Dataset(te_x, label=te_y)
            params = {
                'boosting_type': 'gbdt',
                'objective': 'regression_l2',
                'metric': 'mse',
                'min_child_weight': 1.5,
                'num_leaves': 2**5,
                'lambda_l2': 10,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'colsample_bylevel': 0.7,
                'learning_rate': 0.03,
                'tree_method': 'exact',
                'seed': 2017,
                'nthread': 12,
                'silent': True,
            }
            num_round = 10000
            early_stopping_rounds = 100
            if test_matrix:
                model = clf.train(
                    params,
                    train_matrix,
                    num_round,
                    valid_sets=test_matrix,
                    early_stopping_rounds=early_stopping_rounds)
                pre = model.predict(
                    te_x, 
                    num_iteration=model.best_iteration).reshape(-1, 1)
                train[test_index] = pre
                test_pre[i, :] = model.predict(
                    test_x, 
                    num_iteration=model.best_iteration).reshape(-1, 1)
                cv_scores.append(mean_squared_error(te_y, pre))
        else:
            raise IOError("Please add new clf.")
        print("%s now score is:" % clf_name, cv_scores)
    # 在5折过程中，每一折上模型对test_x预测的平均作为test
    test[:] = test_pre.mean(axis=0)
    # print 模型名称和 5折上各自的mse
    print("%s_score_list:" % clf_name, cv_scores)
    # print 模型名称和5折上平均的mse
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    return train.reshape(-1, 1), test.reshape(-1, 1)


#分步步骤，设定融合模型的基模型

def rf_reg(x_train, y_train, x_valid, kf, label_split=None):
    randomforest = RandomForestRegressor(n_estimators=600,
                                         max_depth=20,
                                         n_jobs=-1,
                                         random_state=2017,
                                         max_features="auto",
                                         verbose=1)
    rf_train, rf_test = stacking_reg(randomforest,
                                     x_train,
                                     y_train,
                                     x_valid,
                                     "rf",
                                     kf,
                                     label_split=label_split)
    return rf_train, rf_test, "rf_reg"


def gb_reg(x_train, y_train, x_valid, kf, label_split=None):
    gbdt = GradientBoostingRegressor(learning_rate=0.04,
                                     n_estimators=100,
                                     subsample=0.8,
                                     random_state=2017,
                                     max_depth=5,
                                     verbose=1)
    gbdt_train, gbdt_test = stacking_reg(gbdt,
                                         x_train,
                                         y_train,
                                         x_valid,
                                         "gb",
                                         kf,
                                         label_split=label_split)
    return gbdt_train, gbdt_test, "gb_reg"


def lgb_reg(x_train, y_train, x_valid, kf, label_split=None):
    lgb_train, lgb_test = stacking_reg(lightgbm,
                                       x_train,
                                       y_train,
                                       x_valid,
                                       "lgb",
                                       kf,
                                       label_split=label_split)
    return lgb_train, lgb_test, "lgb_reg"


#最终版本
#输入是：训练集数据，训练集标签，测试集/验证集数据，交叉验证的折数(kf)，分类器的列表(clf_list)，(clf_fin)
#返回是：
#作用是：得到组合模型的stacking表示

def stacking_pred(
    x_train,y_train,
    x_valid,kf,
    clf_list,label_split=None,
    clf_fin="lgb",
    if_concat_origin=True):
    
    """
    这个地方我觉得写法又问题，只需要一层循环就可以了，没看懂它两层循环想表达的意思。
    该部分目的是得到各个分类器在训练集上预测的train_data_list，和在测试集上预测的test_data_list
    """
    # 遍历分类器对象
    for k, clf_list in enumerate(clf_list):
        clf_list = [clf_list]
        column_list = []# 用来存放分类器的名称
        train_data_list = [] # 用来存放各个分类器在训练集上的预测结果
        test_data_list = []# 用来存放各个分类器在测试集上的稳定预测（在5折预测基础上平均）结果
        
        # 得到各个分类器的在训练集上的预测结果，和在测试集上的稳定预测结果以及分类器的名称
        for clf in clf_list:
            train_data, test_data, clf_name = clf(
                x_train,
                y_train,
                x_valid,
                kf,
                label_split=label_split)
            
            train_data_list.append(train_data)
            test_data_list.append(test_data)
            column_list.append("clf_%s" % (clf_name))
    
    # 将train_data_list与test_data_list的对象横行拼接，即增加列。
    train = np.concatenate(train_data_list, axis=1)
    test = np.concatenate(test_data_list, axis=1)
    
    # 如果为true，即连接原训练集和测试集。
    # 举个例子：比如训练集数据的shape为200x10，即200条数据，10列，训练集的label形状为200x1。
    # 若为5折交叉验证，这个地方拼接上原数据后的train对象的形状就为 200x(10+5) = 200x15
    if if_concat_origin:
        train = np.concatenate([x_train, train], axis=1)
        test = np.concatenate([x_valid, test], axis=1)
    print(x_train.shape)
    print(train.shape)
    print(clf_name)
    print(clf_name in ["lgb"])
    
    """
    下面的是选择最终分类器在拼接后的训练集上训练，在拼接后的测试集上预测。
    """
    if clf_fin in ["rf", "ada", "gb", "et", "lr", "lsvc", "knn"]:
        if clf_fin in ["rf"]:
            clf = RandomForestRegressor(
                n_estimators=600,
                max_depth=20,
                n_jobs=-1,
                random_state=2017,
                max_features="auto",
                verbose=1)
        elif clf_fin in ["gb"]:
            clf = GradientBoostingRegressor(
                learning_rate=0.04,
                n_estimators=100,
                subsample=0.8,
                random_state=2017,
                max_depth=5,
                verbose=1)
        clf.fit(train, y_train)
        pre = clf.predict(test).reshape(-1, 1)
        return pred

    elif clf_fin in ["lgb"]:
        print(clf_name)
        clf = lightgbm
        train_matrix = clf.Dataset(train, label=y_train)
        test_matrix = clf.Dataset(train, label=y_train)
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression_l2',
            'metric': 'mse',
            'min_child_weight': 1.5,
            'num_leaves': 2**5,
            'lambda_l2': 10,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'colsample_bylevel': 0.7,
            'learning_rate': 0.03,
            'tree_method': 'exact',
            'seed': 2017,
            'nthread': 12,
            'silent': True,
        }
        num_round = 10000
        early_stopping_rounds = 100
        model = clf.train(
            params,
            train_matrix,
            num_round,
            valid_sets=test_matrix,
            early_stopping_rounds=early_stopping_rounds)
        print('pred')
        pre = model.predict(
            test,
            num_iteration=model.best_iteration).reshape(-1, 1)
        print(pre)
        return pre

    
#load_dataset导入处理后的数据
data_train = pd.read_csv('train_after_EDA.csv')
data_test = pd.read_csv('test_after_EDA.csv')

#5折交叉
from sklearn.model_selection import  KFold
folds = 5
seed = 1
kf = KFold(n_splits=5, shuffle=True, random_state=0)
  
# 训练集和测试集数据
x_train = data_train[data_test.columns].values
x_valid = data_test[data_test.columns].values
y_train = data_train['target'].values

# 第一层使用gb_reg和rf_reg进行融合
clf_list = [gb_reg, rf_reg]
# 结果传导给第二层，使用lgb_reg进行预测
pred = stacking_pred(x_train, y_train, x_valid, kf, clf_list, 
                     label_split=None, clf_fin="lgb", if_concat_origin=True)