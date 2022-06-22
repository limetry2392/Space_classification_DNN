#-*-coding:utf-8-*-

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from itertools import cycle
import datetime

####세팅 부분#####
LEARNING_DATA = '2022.06.22(Learning_DATA)_허과장님 지점.csv'
SAVE = '공간분류' + '_' + str(datetime.datetime.now().date()) + '.h5' 
#################


feature_keys = [ 
    "Julian",  #0
    "TIME",    #1
    "PM10",    #2
    "CO2",     #3
    "VOCs",    #4
    "NOISE",   #5
    "TEMP",    #6
    "HMDT",    #7
    "PM2.5",   #8
] 

date_time_key = "DATE"
selected_features = [feature_keys[i] for i in [1, 2, 5, 6, 7, 8]]  ##사용되는 특성

##학습 데이터 전처리
test = pd.read_csv('./학습용 데이터/'+LEARNING_DATA, na_values=['-9999', '0', 'NA'])
test = test.dropna(axis=0)
test[date_time_key] = test.DATE.apply(pd.to_datetime)
features = test[selected_features] 
features.index = test[date_time_key] ##날짜정보 인덱스화


##표준화
def normalize(data): 
    data_mean = data.mean(axis=0) 
    data_std = data.std(axis=0) 
    return (data - data_mean) / data_std
#features = normalize(features)

#features2 = normalize(features.iloc[:, 1:8])

#features = pd.concat([features["Julian"], features["TIME"], features2], axis=1)
#features = pd.concat([features["TIME"], features2], axis=1)

#print(features)

#데이터프레임 행렬로 바꾸기
X = features.iloc[:,:].values
y = test.iloc[:,10].values


##라벨정보 숫자형식으로 변경
encoder =  LabelEncoder()
y1 = encoder.fit_transform(y)
Y = pd.get_dummies(y1).values


##데이터 나누기
X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                    test_size=0.33, 
                                                    random_state=1004) 

##모델 초기화
model = Sequential()

##모델 구성
model.add(Dense(12, input_shape=(6,),activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(3,activation='softmax'))

#inputs = keras.layers.Input(shape=(7,)) 
#out = keras.layers.Dense(64, activation='relu')(inputs) 
#outputs = keras.layers.Dense(3, activation='softmax')(out) 
#model = keras.Model(inputs=inputs, outputs=outputs) 


model.compile(loss='categorical_crossentropy', 
              optimizer='Adam', 
              metrics=['accuracy'])


model.summary()

##학습 중단점 설정
path_checkpoint = SAVE 
es_callback = keras.callbacks.EarlyStopping(monitor="val_accuracy", min_delta=0.1, patience=100) 
modelckpt_callback = keras.callbacks.ModelCheckpoint( 
    monitor="val_accuracy", 
    filepath=path_checkpoint, 
    verbose=1, 
    save_weights_only=True, 
    save_best_only=True, 
) 

##모델 학습
hist = model.fit(X_train, 
                 y_train, 
                 validation_data=(X_test, y_test), 
                 validation_split=0.33,
                 epochs=300, 
                 batch_size=12,
                 callbacks=[es_callback, modelckpt_callback]
                )


##모델 정확도
plt.rc('font', family='Malgun Gothic') 
plt.figure(figsize=(12,8))
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.legend(['훈련 정확도','검증 정확도'])
plt.xlabel('학습 반복 수')
plt.ylabel('정확도')
plt.grid()
plt.show()

loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy = {:.2f}".format(accuracy))



##예측 해보기
y_pred = model.predict(X_test)
y_test_class = np.argmax(y_test,axis=1)
y_pred_class = np.argmax(y_pred,axis=1)


##모델평가

print(classification_report(y_test_class, y_pred_class))


##MAE
MAE = mean_absolute_error(y_test_class, y_pred_class)

##MSE
MSE = mean_squared_error(y_test_class, y_pred_class)

##RMSE
RMSE = np.sqrt(MSE)

##R2
R2 = r2_score(y_test_class, y_pred_class)


print('MAE = ', MAE)
print('MSE = ', MSE)
print('RMSE = ', RMSE)
print('R2 = ', R2)

##오차행렬
cm = (confusion_matrix(y_test_class,y_pred_class))
test = pd.DataFrame(cm)

import seaborn as sns
sns.heatmap(cm, annot = True, fmt = 'd',cmap = 'Blues')
plt.xlabel('예측값')
plt.ylabel('실제값')
plt.xticks([0.5,1.5, 2.5],['APT', 'MS', 'PS'])
plt.yticks([0.5,1.5, 2.5],['APT', 'MS', 'PS'])
plt.show()

##ROC

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure()

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
labels = ['APT', 'MS', 'PS']
for i, color in zip(range(3), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.8f})'
             ''.format(labels[i], round(roc_auc[i], 8)))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


model.save(SAVE)