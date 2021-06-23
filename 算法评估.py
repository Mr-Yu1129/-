# -*- coding: utf-8 -*-

#导入包
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

# modelling
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor


# metric for evaluation 构建评分指标 mse rmse
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

#load_dataset导入处理后的数据
data_train = pd.read_csv('train_after_EDA.csv')
data_test = pd.read_csv('test_after_EDA.csv')

#分割训练集
def get_training_data():
    y = data_train.target
    X = data_train.drop([ "target"], axis=1)
    X_train, X_valid, y_train, y_valid = train_test_split(X,
                                                          y,
                                                          test_size=0.3,
                                                          random_state=100)
    return X_train, X_valid, y_train, y_valid

X_train, X_valid,y_train,y_valid = get_training_data()


def get_trainning_data_omitoutliers():
    y1=y_train
    X1=X_train
    return X1,y1


#采用网格搜索训练模型

def train_model(model, param_grid=[], X=[], y=[], splits=5, repeats=5):

    # get unmodified training data, unless data to use already specified
    if len(y) == 0:
        X, y = get_trainning_data_omitoutliers()
        #poly_trans=PolynomialFeatures(degree=2)
        #X=poly_trans.fit_transform(X)
        #X=MinMaxScaler().fit_transform(X)

    # create cross-validation method 创建交叉验证方法
    rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats)

    # perform a grid search if param_grid given 如果指定了param_grid，则执行网格搜索
    if len(param_grid) > 0:
        # setup grid search parameters
        gsearch = GridSearchCV(model,
                               param_grid,
                               cv=rkfold,
                               scoring="neg_mean_squared_error",
                               verbose=1,
                               return_train_score=True)

        # search the grid
        gsearch.fit(X, y)

        # extract best model from the grid
        model = gsearch.best_estimator_
        best_idx = gsearch.best_index_

        # get cv-scores for best model
        grid_results = pd.DataFrame(gsearch.cv_results_)
        cv_mean = abs(grid_results.loc[best_idx, 'mean_test_score'])
        cv_std = grid_results.loc[best_idx, 'std_test_score']

# no grid search, just cross-val score for given model 没有网格搜索，就给模型的交叉得分
    else:
        grid_results = []
        cv_results = cross_val_score(model,
                                     X,
                                     y,
                                     scoring="neg_mean_squared_error",
                                     cv=rkfold)
        cv_mean = abs(np.mean(cv_results))
        cv_std = np.std(cv_results)

    # combine mean and std cv-score in to a pandas series 结合均值和标准CV得分填入pandas中
    cv_score = pd.Series({'mean': cv_mean, 'std': cv_std})

    # predict y using the fitted model 使用拟合模型预测y
    y_pred = model.predict(X)

    # print stats on model performance 打印有关模型性能的统计信息
    print('----------------------')
    print(model)
    print('----------------------')
    print('score=', model.score(X, y))
    print('rmse=', rmse(y, y_pred))
    print('mse=', mse(y, y_pred))
    print('cross_val: mean=', cv_mean, ', std=', cv_std)

    return model, cv_score, grid_results

# places to store optimal models and scores
opt_models = dict()
score_models = pd.DataFrame(columns=['mean','std'])
# no. k-fold splits
splits=5
# no. k-fold iterations
repeats=5

 #gbdt模型
model = 'GradientBoosting'
opt_models[model] = GradientBoostingRegressor()

param_grid = {
    'n_estimators': [150, 250, 350],
    'max_depth': [1, 2, 3],
    'min_samples_split': [5, 6, 7]
}

opt_models[model], cv_score, grid_results = train_model(
    opt_models[model],
    param_grid=param_grid,
    splits=splits,
    repeats=1)

cv_score.name = model
score_models = score_models.append(cv_score)

#XGB模型
model = 'XGB' 
#opt_models[model] = XGBRegressor()
opt_models[model] = XGBRegressor(objective='reg:squarederror')
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [1, 2, 3],
}

opt_models[model], cv_score, grid_results = train_model(
    opt_models[model],
    param_grid=param_grid,
    splits=splits,
    repeats=1)

cv_score.name = model
score_models = score_models.append(cv_score)

#随机森林模型
model = 'RandomForest' 
opt_models[model] = RandomForestRegressor()

param_grid = {
    'n_estimators': [100, 150, 200],
    'max_features': [8, 12, 16, 20, 24],
    'min_samples_split': [2, 4, 6]
}

opt_models[model], cv_score, grid_results = train_model(
    opt_models[model],
    param_grid=param_grid,
    splits=5,
    repeats=1)

cv_score.name = model
score_models = score_models.append(cv_score)


def model_predict(test_data,test_y=[],stack=False): 
    i=0
    y_predict_total=np.zeros((test_data.shape[0],))
    for model in opt_models.keys():
        if model!="LinearSVR" and model!="KNeighbors":
            y_predict=opt_models[model].predict(test_data)
            y_predict_total+=y_predict
            i+=1
        if len(test_y)>0:
            print("{}_mse:".format(model),mean_squared_error(y_predict,test_y))
    y_predict_mean=np.round(y_predict_total/i,3)
    if len(test_y)>0:
        print("mean_mse:",mean_squared_error(y_predict_mean,test_y))
    else:
        y_predict_mean=pd.Series(y_predict_mean)
        return y_predict_mean

model_predict(X_valid,y_valid)
#接下来改那几个模型，整合一下，不出结果，汇总到最后出结果
