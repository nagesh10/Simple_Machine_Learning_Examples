import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('D:\\Datasets\\node-fussy-examples-master\\node-fussy-examples-master\\sonar\\training.csv')
ds=df.values
x_train=df[df.columns[0:60]].values
y_train=df[df.columns[60]]

encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)

print(x_train.shape[1])
print(encoded_Y.shape)
print(encoded_Y)
model=Sequential()
model.add(Dense(60,activation='relu',input_shape=(x_train.shape[1],)))
model.add(Dense(60,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train, encoded_Y, epochs=100,verbose=1)


df_test=pd.read_csv('D:\\Datasets\\node-fussy-examples-master\\node-fussy-examples-master\\sonar\\testing.csv')
dtest=df_test[df_test.columns[0:60]].values
y_test=df_test[df_test.columns[60]]

encoder = LabelEncoder()
encoder.fit(y_test)
encoded_Ytest = encoder.transform(y_test)
score=model.evaluate(dtest,encoded_Ytest,verbose=0)
print('loss: ', score[0])
print('Acuraccy: ', score[1])
test_y_predict=model.predict(dtest)
temp=test_y_predict.flatten()
print(temp)

label=[]
for i in temp:
    if i>0.5:
        label.append('R')
    else:
        label.append('M')
print(label)
# print((test_y_predict>0.5)*1)S