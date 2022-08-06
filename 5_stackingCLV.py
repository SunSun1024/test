from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingCVClassifier
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import warnings
import time
import math
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

warnings.filterwarnings(action ='ignore',category = FutureWarning)
warnings.filterwarnings(action ='ignore',category = DeprecationWarning)
warnings.filterwarnings(action ='ignore',category = UserWarning)

# 获取数据集
#iris = datasets.load_iris()
#X, y = iris.data[:, 1:3], iris.target

feature_file = pd.read_csv("UNSW_NB15_training-set.csv")

x = []# 特征数据
y = []# 标签
for index in feature_file.index.values:
    # print('index', index)
    # print(feature_file.ix[index].values)
    x.append(feature_file.iloc[index].values[1: -1]) # 每一行都是ID+特征+Label
    y.append(feature_file.iloc[index].values[-1] - 1) #
x, y = np.array(x), np.array(y)
# print(y)
print('x,y shape', np.array(x).shape, np.array(y).shape)
print('样本数', len(feature_file.index.values))
# 分训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=12343)
print('训练集和测试集 shape', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)

RANDOM_SEED = 42
clf1 = XGBClassifier()
clf2 = RandomForestClassifier(random_state=RANDOM_SEED)
clf3 = DecisionTreeClassifier()
lr = LogisticRegression()

# Stacking集成
sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3],
                            use_probas=True,
                            meta_classifier=lr,
                            random_state=RANDOM_SEED)
start = time.time()

params = {'randomforestclassifier__n_estimators': [10, 50],
          'meta_classifier__C': [0.1, 10.0]}

grid = GridSearchCV(estimator=sclf,
                    param_grid=params,
                    cv=5,
                    refit=True)
grid.fit(x, y)

cv_keys = ('mean_test_score', 'std_test_score', 'params')

for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    print("%0.3f +/- %0.2f %r"
          % (grid.cv_results_[cv_keys[0]][r],
             grid.cv_results_[cv_keys[1]][r] / 2.0,
             grid.cv_results_[cv_keys[2]][r]))

print('Best parameters: %s' % grid.best_params_)
print('Accuracy: %.2f' % grid.best_score_)
print("time:",time_since(start))


# 输出其他指标
sclf.fit(X_train, y_train)
y_pre = sclf.predict(X_test)
recall = recall_score(y_test, y_pre, average='macro')
precision = precision_score(y_test, y_pre, average='macro')
f1 = f1_score(y_test, y_pre, average='macro')
c = confusion_matrix(y_test, y_pre)
print("StackingCVClassifier指标：")
# print("acc:", acc)
print("recall:", recall)
print("precision:", precision)
print("f1:", f1)
print("c:", c)
print("time:",time_since(start))


