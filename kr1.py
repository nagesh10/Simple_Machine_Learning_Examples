import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

trX=np.linspace(-1,1,101)
trY=2*trX+np.random.randn(trX.shape[0])*0.33

model=Sequential()
model.add(Dense(1,activation='linear',input_dim=1))
model.compile(optimizer='sgd',loss='mse')
model.summary()
weights = model.layers[0].get_weights()
w_init = weights[0][0][0]
b_init = weights[1][0]

print('Linear regression model is initialized with weight w: %.2f, b: %.2f' % (w_init, b_init))
model.fit(trX,trY,epochs=100,verbose=1)

weights = model.layers[0].get_weights()
w = weights[0][0][0]
b = weights[1][0]
print('Linear regression model is trained with weight w: %.2f, b: %.2f' % (w, b))

plt.plot(trX, trY, label='data')
plt.plot(trX, w_init*trX + b_init, label='init')
plt.plot(trX, w*trX + b, label='prediction')
plt.legend()
plt.show()