import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

np.random.seed(1337)
X=np.linspace(-1,1,200)
np.random.shuffle(X)
Y=0.5*X+2+np.random.normal(0,0.05,(200,))
#plt.scatter(X,Y)
#plt.show()

x_train=X[:160]
y_train=Y[:160]
x_test=X[160:]
y_test=Y[160:]



model=Sequential()
model.add(Dense(output_dim=1,input_dim=1))
W,b=model.get_weights()
print('Weights at beginning: ', W,' Bias: ',b)
model.summary()

model.compile(loss='mse',optimizer='sgd',metrics=['accuracy'])

print('........Training.........')
for i in range(301):
    #cost = model.train_on_batch(x_train, y_train)
    cost = model.fit(x_test, y_test, 20, verbose=1)
    if i%100==0:
        print('Step: ',i,' cost is : ', cost.history['loss'])
#print(cost.history['loss'])
print(cost.history['acc'])

print('.........Testing...........')
cost=model.evaluate(x_test,y_test,20,verbose=1)
print('cost at testing: ', cost)

W,b=model.get_weights()
print('Weights: ', W,' Bias: ',b)


