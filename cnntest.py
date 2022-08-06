#前面数据处理都是一样的
import pandas as pd
import numpy as np
from tensorflow.python.eager.monitoring import Metric
df = pd.read_csv('attack_classification.csv')
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
#train_inf = np.isinf(df)
#df[train_inf] = 0


#df = pd.read_csv('datas/dataset.csv')
target=df['label'] #仅最后一列
data=df.drop('label',axis=1)
#data=df.iloc[:,[0,1,2,3,21,23,70,69,6,13,66,15,12]]#前12个特征
#data=df.iloc[:,[2,1,70,3,69,0,21,11,26,56,23,12,6]]
feature_names=data.columns
data.head() #去掉标签的特征集

#create a train test split
import matplotlib.pyplot as plt #可视化
import seaborn as sns #可视化
import warnings
warnings.filterwarnings('ignore')
import io
# import plotly.offline as py
# import plotly.graph_objects as go
# import plotly.tools as tls
# import plotly.figure_factory as ff
# from plotly.offline import iplot
# import plotly.graph_objects as go

lab=df['label'].value_counts().keys().tolist()
val=df['label'].value_counts().values.tolist()
print(lab)
print(val)

#pre-processing,看看分布
y=df['label'].values.reshape(-1,1)
target_names=['0','1','2','3','4','5','6','7']

#使用sklearn进行预处理
from sklearn.preprocessing import LabelEncoder

label_encoder=LabelEncoder()
label_encoder.fit(y)
encoded_y=label_encoder.transform(y)
print(encoded_y)
from sklearn.model_selection import train_test_split
import numpy as np
X_train,X_test,y_train,y_test=train_test_split(data,encoded_y,random_state=np.random)
X_train.head()
print(X_train)
#标准化处理？
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_scaler=scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
print(X_train_scaled)
print(X_test_scaled)


import tensorflow as tf
import glob
from  functools import partial
from keras.models import Sequential
from keras.layers import Flatten,Conv1D,MaxPooling1D,Dropout,Dense
from tensorflow.keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import keras

num_classes = 15
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
Y_train_cnn = to_categorical(y_train, num_classes)
Y_test_cnn = to_categorical(y_test, num_classes)
print(Y_train_cnn.shape,Y_test_cnn.shape,X_train_cnn.shape,X_test_cnn.shape)
#activation='relu'
# from kerastuner.tuners import RandomSearch
# def build_model(hp):
#     model = Sequential()
#     model.add(Conv1D(128, strides=3, input_shape=X_train_cnn.shape[1:], activation=activation, kernel_size=4, padding='same'))
#     #kernel_size 卷积核大小
#     model.add(MaxPooling1D())
#     model.add(Conv1D(64, strides=1, activation=activation, kernel_size=5, padding='same'))
#     model.add(MaxPooling1D())
#     model.add(Flatten())
#     model.add(Dense(units=hp.Int('units',
#                                 min_value=32,
#                                 max_value=512,
#                                 step=32)
#     ,input_dim = 8))#输出的标签大小嘛
#     model.add(Dropout(0.2))
#     model.add(Dense(100, activation=activation))
#     model.add(Dropout(0.2))
#     model.add(Dense(50, activation=activation))
#     model.add(Dropout(0.2))
#     model.add(Dense(num_classes, activation='softmax'))
#     print(model.summary())
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model

# tuner = RandomSearch(
#     build_model,
#     objective='val_accuracy',  #优化目标为精度'val_accuracy'（最小化目标）
#     max_trials=5,   #总共试验5次，选五个参数配置
#     executions_per_trial=1, #每次试验训练模型三次
#     overwrite=True,
#     directory="tuner_random_directory",
#     project_name="tuner_random_project_name",
#     )
# tuner.search_space_summary()
# tuner.search(X_train_cnn, Y_train_cnn,epochs=10, batch_size=5000,validation_data=(X_test_cnn, Y_test_cnn))
# models = tuner.get_best_models(num_models=2)
# tuner.results_summary()




activation='relu'
model = Sequential()
model.add(Conv1D(128, strides=3, input_shape=X_train_cnn.shape[1:], activation=activation, kernel_size=4, padding='same'))
#kernel_size 卷积核大小，奇数5
model.add(MaxPooling1D())
model.add(Conv1D(64, strides=1, activation=activation, kernel_size=5, padding='same'))
#卷积太少了，全连接2-3层
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(1024,input_dim = 8))#输出的标签大小嘛
model.add(Dropout(0.2))
#model.add(Dense(1024, activation=activation))
#model.add(Dropout(0.2))
#model.add(Dense(1024, activation=activation))
#model.add(Dropout(0.2))# model.add(Dense(1024, activation=activation))
model.add(Dropout(0.5))
model.add(Dense(1024, activation=activation))
model.add(Dropout(0.5))
model.add(Dense(756, activation=activation))
model.add(Dropout(0.5))
#el.add(Dense(512, activation=activation))
#odel.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

result = model.fit(X_train_cnn, Y_train_cnn, verbose=1, epochs=50, batch_size=400,
                   validation_data=(X_test_cnn, Y_test_cnn))

print("Length of Train Data:", len(X_train_cnn))
print("Length of Test Data:", len(X_test_cnn))

from sklearn.metrics import confusion_matrix
print("For Test Data: ")
Y_pred = model.predict(X_test_cnn)
print("Confusion Matrix:")
matrix = confusion_matrix(Y_test_cnn.argmax(axis=1), Y_pred.argmax(axis=1))
for i in range(len(matrix)):
    k = matrix[i,:]
    for j in k:
        print(j,end=" ")
    print("")
for i in range(num_classes):
    print( target_names[i],": ",(matrix[i,i]/sum(matrix[i,:]))*100,"%")
matrix = pd.DataFrame(matrix, index =target_names ,columns = target_names)
plt.figure(figsize = (10,7))
#sn.heatmap(matrix, annot=True,annot_kws={"size": 8})

from sklearn.metrics import classification_report
print("For Test Data: Full Classification Report ")
Y_test = np.argmax(Y_test_cnn, axis=1) # Convert one-hot to index
y_pred = model.predict_classes(X_test_cnn)   #2.5版本
#y_pred=model.predict(X_test_cnn)
#Y_test = np.argmax(Y_test_cnn, axis=1) # Convert one-hot to index
#classes_x=np.argmax(y_pred,axis=1)
print(classification_report(Y_test, y_pred))




# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense,Dropout
# from keras.layers import Conv1D,GlobalAveragePooling1D,MaxPooling1D
# import keras.metrics
# from sklearn.model_selection import train_test_split
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import Normalizer
# import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import f1_score

# sampled_data = pd.read_csv('datanew/data_8class.csv')
# sampled_X = sampled_data.iloc[:,:81]#特征
# print(sampled_X)
# ys = sampled_data['ProtocolName']#最后一个为类别
# print(ys)
# from sklearn.preprocessing import LabelEncoder
# label_encoder=LabelEncoder()
# label_encoder.fit(ys)
# sampled_y=label_encoder.transform(ys)
# print(sampled_y)
# #分裂
# train,test,train_label,test_label=train_test_split(
#     sampled_X,sampled_y,test_size=0.2,random_state=42)
# #归一化处理
# scaler=Normalizer().fit(train)
# train=scaler.transform(train)

# scaler=Normalizer().fit(test)
# test=scaler.transform(test)

# train=np.expand_dims(train,axis=2)
# test=np.expand_dims(test,axis=2)
# print(train)

#标签数组化
# train_label=np.reshape(train_label,train_label.shape[0])
# test_label=np.reshape(test_label,test_label.shape[0])
# print(train_label)
# cnn_1D=Sequential()
# cnn_1D.add(Conv1D(512,1,activation='relu',input_shape=(81,1)))
# cnn_1D.add(Conv1D(256,1,activation='relu'))
# cnn_1D.add(MaxPooling1D(3))
# cnn_1D.add(Conv1D(64,1,activation='relu'))
# cnn_1D.add(Conv1D(64,1,activation='relu'))
# cnn_1D.add(GlobalAveragePooling1D())
# cnn_1D.add(Dropout(0.2))
# cnn_1D.add(Dense(8,activation='softmax'))
# cnn_1D.compile(
#     loss='sparse_categorical_crossentropy',
#     optimizer='rmsprop',
#     metrics=(['accuracy'])
# )

# print(cnn_1D.summary())

# #模型拟合
# history=cnn_1D.fit(train,train_label,batch_size=400,epochs=50,validation_data=(test,test_label))
# #预测，并得出准确度分数
# score=cnn_1D.evaluate(test,test_label,batch_size=400)
# print(score)
# y=cnn_1D.predict(test)

# print("————————训练图形绘制——————————")

# # list all data in history
# print(history.history.keys())
# # summarize history for accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train','test'], loc='upper left')
# plt.show()

# # summarize history for loss

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train','test'], loc='upper left')
# plt.show()