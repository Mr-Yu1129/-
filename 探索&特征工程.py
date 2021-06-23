# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})#matplotlib避免报错
import seaborn as sns#可视化包
import pandas as pd
import numpy as np
from scipy import stats #功能，提供t检验，正态性检验，卡方检验之类

#load_dataset导入数据
train_data_file = "./zhengqi_train.txt"
test_data_file =  "./zhengqi_test.txt"
data_train = pd.read_csv(train_data_file, sep='\t', encoding='utf-8')
data_test  = pd.read_csv(test_data_file, sep='\t', encoding='utf-8')

#绘制各个特征的箱线图
plt.figure(figsize=(18, 10))
plt.boxplot(x=data_train.values,labels=data_train.columns)
plt.hlines([-7.5, 7.5], 0, 40, colors='r')
plt.show()

#全部变量的KDE分布图
dist_cols = 6
dist_rows = len(data_test.columns)
plt.figure(figsize=(4 * dist_cols, 4 * dist_rows))
i = 1
for col in data_test.columns:
    ax = plt.subplot(dist_rows, dist_cols, i)
    ax = sns.kdeplot(data_train[col], color="Red", shade=True)
    ax = sns.kdeplot(data_test[col], color="Blue", shade=True)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax = ax.legend(["train", "test"])
    i += 1
plt.show()
'''#画出训练集与测试集分布不一致的特征
drop_col = 6
drop_row = 1
plt.figure(figsize=(5 * drop_col, 5 * drop_row))
for i, col in enumerate(["V5", "V9", "V11", "V17", "V22", "V28"]):
    ax = plt.subplot(drop_row, drop_col, i + 1)
    ax = sns.kdeplot(data_train[col], color="Red", shade=True)
    ax = sns.kdeplot(data_test[col], color="Blue", shade=True)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax = ax.legend(["train", "test"])
plt.show()
'''
#热力图相关性
plt.figure(figsize=(20, 16))  
column = data_train.columns.tolist()  
mcorr = data_train[column].corr(method="spearman")  
mask = np.zeros_like(mcorr, dtype=np.bool)  
mask[np.triu_indices_from(mask)] = True  
cmap = sns.diverging_palette(220, 20, as_cmap=True)
g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, 
                annot=True, fmt='0.2f',annot_kws={'size': 6})
plt.show()

#merge train_set and test_set合并数据
data_train["oringin"]="train"
data_test["oringin"]="test"
data_all=pd.concat([data_train,data_test],axis=0,ignore_index=True)
#删除训练集测试集分布不一致的特征
data_all.drop(["V5","V9","V11","V17","V22","V28"],axis=1,inplace=True)

# normalise numeric columns数据归一化
cols_numeric=list(data_all.columns)
cols_numeric.remove("oringin")
def scale_minmax(col):
    return (col-col.min())/(col.max()-col.min())
scale_cols = [col for col in cols_numeric if col!='target']
data_all[scale_cols] = data_all[scale_cols].apply(scale_minmax,axis=0)

#Check effect of Box-Cox transforms on distributions of continuous variables
#处理好的数据通过直方图、Q_Q图、线性关系图展示
fcols = 6
frows = len(cols_numeric)-1
plt.figure(figsize=(4*fcols,4*frows))
i=0

for var in cols_numeric:
    if var!='target':
        dat = data_all[[var, 'target']].dropna()
        
        i+=1
        plt.subplot(frows,fcols,i)
        sns.distplot(dat[var] , fit=stats.norm);
        plt.title(var+' Original')
        plt.xlabel('')
        
        i+=1
        plt.subplot(frows,fcols,i)
        _=stats.probplot(dat[var], plot=plt)
        plt.title('skew='+'{:.4f}'.format(stats.skew(dat[var])))
        plt.xlabel('')
        plt.ylabel('')
        
        i+=1
        plt.subplot(frows,fcols,i)
        plt.plot(dat[var], dat['target'],'.',alpha=0.5)
        plt.title('corr='+'{:.2f}'.format(np.corrcoef(dat[var], dat['target'])[0][1]))
 
        i+=1
        plt.subplot(frows,fcols,i)
        trans_var, lambda_var = stats.boxcox(dat[var].dropna()+1)
        trans_var = scale_minmax(trans_var)      
        sns.distplot(trans_var , fit=stats.norm);
        plt.title(var+' Tramsformed')
        plt.xlabel('')
        
        i+=1
        plt.subplot(frows,fcols,i)
        _=stats.probplot(trans_var, plot=plt)
        plt.title('skew='+'{:.4f}'.format(stats.skew(trans_var)))
        plt.xlabel('')
        plt.ylabel('')
        
        i+=1
        plt.subplot(frows,fcols,i)
        plt.plot(trans_var, dat['target'],'.',alpha=0.5)
        plt.title('corr='+'{:.2f}'.format(np.corrcoef(trans_var,dat['target'])[0][1]))

#Box-Cox变换，使其满足正态性       
cols_transform=data_all.columns[0:-2]
for col in cols_transform:   
    # transform column
    data_all.loc[:,col], _ = stats.boxcox(data_all.loc[:,col]+1)    

#特征进行对数变换，使数据更符合正态
sp = data_train.target
data_train.target1 =np.power(1.5,sp)
print(data_train.target1.describe())

# function to get training samples分割合并的训练集与测试集
df_train = data_all[data_all["oringin"] == "train"].reset_index(drop=True)
df_train.drop(["oringin"],axis=1,inplace=True)
df_test = data_all[data_all["oringin"] == "test"].reset_index(drop=True)
df_test.drop(["oringin", "target"], axis=1,inplace=True)

#PCA跟构建新特征、方向上相反，所以具体用哪一个得看跑出来的结果再调整
'''
#PCA处理 
from sklearn.decomposition import PCA   #主成分分析法
pca = PCA(n_components=0.9)   #保持90%的信息 但是偶尔会报错，可以改成保留多少个特征
new_train_pca_90 = pca.fit_transform(df_train.iloc[:,0:-1])
new_test_pca_90 = pca.transform(df_test)
new_train_pca_90 = pd.DataFrame(new_train_pca_90)
new_test_pca_90 = pd.DataFrame(new_test_pca_90)
new_train_pca_90['target'] = df_train['target']
new_train_pca_90.describe()


#构建新特征  ~but这里构建后的新特征太多了，而且因为特征没有字面意义，不能够重点处理
epsilon = 1e-5
#组交叉特征，可以自行定义，如增加： x*x/y, log(x)/y 等等
func_dict = {
    'add': lambda x, y: x + y,
    'mins': lambda x, y: x - y,
    'div': lambda x, y: x / (y + epsilon),
    'multi': lambda x, y: x * y
}

def auto_features_make(train_data, test_data, func_dict, col_list):#双循环
    train_data, test_data = train_data.copy(), test_data.copy()
    for col_i in col_list: #1
        for col_j in col_list:#2
            for func_name, func in func_dict.items():#1
                for data in [train_data, test_data]:#2
                    func_features = func(data[col_i], data[col_j])
                    col_func_features = '-'.join([col_i, func_name, col_j])
                    data[col_func_features] = func_features
    return train_data, test_data

train_data2, test_data2 = auto_features_make(
    df_train,df_test,func_dict,col_list=df_test.columns)

'''
##保存数据
df_train.to_csv('train_after_EDA.csv', float_format='%.6f', index=False)
df_test.to_csv('test_after_EDA.csv', float_format='%.6f', index=False)