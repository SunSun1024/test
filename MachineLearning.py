#import xgboost as xgb
#from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import warnings

from sklearn.preprocessing import LabelEncoder


#from xgboost.sklearn import XGBClassifier
from sklearn import metrics
warnings.filterwarnings(action ='ignore',category = FutureWarning)
warnings.filterwarnings(action ='ignore',category = DeprecationWarning)
warnings.filterwarnings(action ='ignore',category = UserWarning)
# warnings.filterwarnings(action ='ignore',category = Warning)

# warnings.filterwarnings(action ='ignore',category = ConvergenceWarning)
# 加载样本数据集
# feature_file = pd.read_csv("unusual_dataset222.csv")

df = pd.read_csv('CNN/attack_classification.csv')
# 读取数据
#train = pd.read_csv('./data/train.csv')

# 检查数据中是否有缺失值，以下两种方式均可
# Flase:对应特征的特征值中无缺失值
# True：有缺失值
print(df.isnull().any())
#print(np.isnan(df).any())

# 查看缺失值记录
train_null = pd.isnull(df)
train_null = df[train_null == True]
print(train_null)

# 缺失值处理，以下两种方式均可
# 删除包含缺失值的行
df.dropna(inplace=True)
# 缺失值填充
df.fillna('100')



#数据处理
train_inf = np.isinf(df)
df[train_inf] = 0

x = []# 特征数据
y = []# 标签
print(df.index.values)
for index in df.index.values:
    # print('index', index)
    # print(feature_file.ix[index].values) 
    x.append(df.iloc[index].values[0 : -1]-1) # 每一行都是ID+特征+Label
    y.append(df.iloc[index].values[-1] - 1) #
x, y = np.array(x), np.array(y)
# print(y)
print('x,y shape', np.array(x).shape, np.array(y).shape)
print('样本数', len(df.index.values))
# 分训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(x, y, test_size=0.3, stratify = y)
le = LabelEncoder()
y_train = le.fit_transform(y_train)
print('训练集和测试集 shape', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# model = xgb.XGBRegressor(max_depth=6,learning_rate=0.05,n_estimators=100,randam_state=42)
# model.fit(x,y)
# y_pre=model.predict(y)


Cross_Validation = True

##############################模型
# xgboost
#from xgboost import XGBClassifier
#xgbc_model=XGBClassifier()

# 随机森林
from sklearn.ensemble import RandomForestClassifier
rfc_model=RandomForestClassifier()

# ET
from sklearn.ensemble import ExtraTreesClassifier
et_model=ExtraTreesClassifier()

# 朴素贝叶斯
from sklearn.naive_bayes import GaussianNB
gnb_model=GaussianNB()

#K最近邻
from sklearn.neighbors import KNeighborsClassifier
knn_model=KNeighborsClassifier()

#逻辑回归
from sklearn.linear_model import LogisticRegression
lr_model=LogisticRegression()

#决策树
from sklearn.tree import DecisionTreeClassifier
dt_model=DecisionTreeClassifier()

#支持向量机
from sklearn.svm import SVC
svc_model=SVC()

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score


if Cross_Validation:
    # xgboost
    #xgbc_model.fit(X_train,y_train)
    #y_pre = xgbc_model.predict(X_test)
    ''''
    acc = cross_val_score(xgbc_model,x,y,cv=5).mean()
    recall = recall_score(y_test, y_pre, average='macro')
    precision = precision_score(y_test, y_pre, average='macro')
    f1 = f1_score(y_test, y_pre, average='macro')
    c = confusion_matrix(y_test, y_pre)
    print("xgBoost模型：")
    print("acc:", acc)
    print("recall:", recall)
    print("precision:", precision)
    print("f1:", f1)
    print("c:", c)
    '''''

    # 随机森林
    rfc_model.fit(X_train,y_train)
    y_pre = rfc_model.predict(X_test)

    acc = cross_val_score(rfc_model,x,y,cv=5).mean()
    recall = recall_score(y_test, y_pre, average='macro')
    precision = precision_score(y_test, y_pre, average='macro')
    f1 = f1_score(y_test, y_pre, average='macro')
    c = confusion_matrix(y_test, y_pre)
    print("随机森林模型：")
    print("acc:", acc)
    print("recall:", recall)
    print("precision:", precision)
    print("f1:", f1)
    print("c:", c)

    # ET
    # et_model.fit(x,y)

    # 朴素贝叶斯
    gnb_model.fit(X_train,y_train)
    y_pre = gnb_model.predict(X_test)

    acc = cross_val_score(gnb_model,x,y,cv=5).mean()
    recall = recall_score(y_test, y_pre, average='macro')
    precision = precision_score(y_test, y_pre, average='macro')
    f1 = f1_score(y_test, y_pre, average='macro')
    c = confusion_matrix(y_test, y_pre)
    print("朴素贝叶斯模型：")
    print("acc:", acc)
    print("recall:", recall)
    print("precision:", precision)
    print("f1:", f1)
    print("c:", c)
    
    # K最近邻
    knn_model.fit(X_train,y_train)
    y_pre = knn_model.predict(X_test)

    acc = cross_val_score(knn_model,x,y,cv=5).mean()
    recall = recall_score(y_test, y_pre, average='macro')
    precision = precision_score(y_test, y_pre, average='macro')
    f1 = f1_score(y_test, y_pre, average='macro')
    c = confusion_matrix(y_test, y_pre)
    print("knn模型：")
    print("acc:", acc)
    print("recall:", recall)
    print("precision:", precision)
    print("f1:", f1)
    print("c:", c)
    
    # 逻辑回归
    # lr_model.fit(x,y)
    
    # 决策树
    dt_model.fit(X_train,y_train)
    y_pre = dt_model.predict(X_test)

    acc = cross_val_score(dt_model,x,y,cv=5).mean()
    recall = recall_score(y_test, y_pre, average='macro')
    precision = precision_score(y_test, y_pre, average='macro')
    f1 = f1_score(y_test, y_pre, average='macro')
    c = confusion_matrix(y_test, y_pre)
    print("决策树模型：")
    print("acc:", acc)
    print("recall:", recall)
    print("precision:", precision)
    print("f1:", f1)
    print("c:", c)
    
    # 支持向量机
    svc_model.fit(X_train,y_train)
    y_pre = svc_model.predict(X_test)

    acc = cross_val_score(svc_model,x,y,cv=5).mean()
    recall = recall_score(y_test, y_pre, average='macro')
    precision = precision_score(y_test, y_pre, average='macro')
    f1 = f1_score(y_test, y_pre, average='macro')
    c = confusion_matrix(y_test, y_pre)
    print("SVM模型：")
    print("acc:", acc)
    print("recall:", recall)
    print("precision:", precision)

    print("f1:", f1)
    print("c:", c)

    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score


    print("\n使用５折交叉验证方法得随机森林模型的准确率（每次迭代的准确率的均值）：")
    #print("\tXGBoost模型：",cross_val_score(xgbc_model,x,y,cv=5).mean())
    print("\t随机森林模型：",cross_val_score(rfc_model,x,y,cv=5).mean())
    # print("\tET模型：",cross_val_score(et_model,x,y,cv=5).mean())
    print("\t高斯朴素贝叶斯模型：",cross_val_score(gnb_model,x,y,cv=5).mean())
    print("\tK最近邻模型：",cross_val_score(knn_model,x,y,cv=5).mean())
    # print("\t逻辑回归：",cross_val_score(lr_model,x,y,cv=5).mean())
    print("\t决策树：",cross_val_score(dt_model,x,y,cv=5).mean())
    print("\t支持向量机：",cross_val_score(svc_model,x,y,cv=5).mean())


# 使用交叉验证在Xgboost、随机森林、ET、朴素贝叶斯模型的准确率

