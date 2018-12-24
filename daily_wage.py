import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',15)
df_train=pd.read_csv('D:\Datasets\hourly_wages_data.csv')

train_X=df_train.drop(columns=['wage_per_hour'])
train_Y=df_train[['wage_per_hour']]

model=Sequential()
n_cols=train_X.shape[1]
print(train_X.shape)
model.add(Dense(50,activation='relu',input_shape=(n_cols,)))
model.add(Dense(50,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(train_X,train_Y,validation_split=0.2,epochs=30)


df_test=pd.read_csv('D:\Datasets\hwd_test.csv')
test_y_predict=model.predict(df_test)
print(test_y_predict)
