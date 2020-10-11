from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, Conv2D, Flatten, MaxPool2D
from mlxtend.data import loadlocal_mnist
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
import keras
from tensorflow.python.client import device_lib
import time
import os

tf.disable_v2_behavior()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# comment out to use GPU, and uncomment to use CPU. Use for bench marking only!
# vvv

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print('\n---\n')

if tf.test.gpu_device_name():
    print('GPU found!')
else:
    print("No GPU found!")

print('\n[console] Available devices: \n', device_lib.list_local_devices())

X, y = loadlocal_mnist(
  images_path='train-images-idx3-ubyte',
  labels_path='train-labels-idx1-ubyte')

print('=====================\n[console] PREPARING TRAINING DATA...')

# prepare the data in a 2D format

X = np.reshape(X, (len(X), 28, 28, 1))

print(X.shape)

enc_y = to_categorical(y)

tX, ty = loadlocal_mnist(
  images_path='t10k-images-idx3-ubyte',
  labels_path='t10k-labels-idx1-ubyte')

print('[console] PREPARING TEST DATA...')

tX = np.reshape(tX, (len(tX), 28, 28, 1))
enc_ty = to_categorical(ty)

# initizalize model

model = Sequential([ #conv net just for a bit more torture
  Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)),
  Conv2D(32, (3,3), activation='relu'),
  Conv2D(8, (3,3), activation='selu'),
  MaxPool2D(pool_size=[14,14], strides=[2,2]),
  Flatten()
])

# tracking data
time_prd = []
acc = []
loss = []
ce_loss = []

epoch = 20
lr = 0.001
b_size = 50

# the making of a horrible MLP

model.add(Dense(100, activation='relu'))
model.add(Dense(600, activation='selu')) # these layers don't do much because of the vanishing gradient. It just makes the computation longer and more torturous.
model.add(Dense(300, activation='selu')) # the actual decision part start here)
model.add(Dense(300, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(10, activation='softmax')) # soft max for probability

opt = Adam(lr = lr, decay = lr/epoch)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics = ['accuracy', 'binary_crossentropy', 'mean_squared_error'])

# start training
start_time = time.time()

for i in range(epoch):
  print('total_epochs: ', i+1, '/', epoch, '\n---')
  model.fit(X, enc_y, epochs = 1, batch_size = b_size, validation_split = 0.005, shuffle = True)
  # store current data
  dat = model.evaluate(tX, enc_ty)
  accd = dat[1]
  losd = dat[3]
  ce_lossd = dat[2]
  time_prd.append(i+1)
  acc.append(accd)
  loss.append(losd)
  ce_loss.append(ce_lossd)
  print('\n')

print('\n---\n')

end_time = time.time()

predictions = model.predict(tX)
answers = list(zip(ty[:20],[np.argmax(x) for x in predictions[:20]]))

print("answer | predic")

for i in answers:
  print(i)

print('\n---\n')

accuracy = model.evaluate(tX, enc_ty)[1]
print('accuracy: %', accuracy*100)
print('training_time: ', (end_time - start_time), " seconds")

# export the network for high accuracy MNIST training
if os.path.isfile('models/MNISTnet.h5'):
  os.remove('models/MNISTnet.h5')

model.save('models/MNISTnet.h5')

# show the plot
# plt.plot(time_prd,acc)
# plt.plot(time_prd, loss)
# plt.plot(time_prd, ce_loss)
# plt.legend(['Accuracy', 'Mean Absolute Error', 'Crossentropy Loss'], loc=4)
# plt.plot([0, max(time_prd)], [1, 1])
# plt.show()
