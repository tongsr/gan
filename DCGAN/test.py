import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential

x = np.array(range(0, 10000),dtype=float)
y = x*3+2

model = Sequential([
    Dense(8, input_shape=[1], activation='relu'),
    Dense(1)
])

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, verbose=1,epochs = 100,batch_size=30)


result = model.predict([3])
print(result)