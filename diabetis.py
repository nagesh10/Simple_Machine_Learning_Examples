from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from keras.utils import to_categorical
pd.set_option('display.max_columns',15)

train_df_2 = pd.read_csv('D:\Datasets\diabetes_data.csv')
train_X_2 = train_df_2.drop(columns=['diabetes'])
print(train_X_2.head())
n_cols=train_X_2.shape[1]
train_Y_2=to_categorical(train_df_2.diabetes)
print(train_Y_2[0:5])

model=Sequential()
model.add(Dense(250,activation='relu',input_shape=(n_cols,)))
model.add(Dense(250,activation='relu'))
model.add(Dense(250,activation='relu'))
model.add(Dense(2,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_X_2, train_Y_2, epochs=30, validation_split=0.2)

df_test=pd.read_csv('D:\Datasets\dbt_test.csv')
test_y_predict=model.predict(df_test)
print(test_y_predict)