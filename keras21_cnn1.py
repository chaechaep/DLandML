from keras.models import Sequential

filter_size = 43
kernel_size = (3, 3)

from keras.layers import Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(7, (2, 2), padding='same',
                 input_shape=(10, 10, 1)))

model.add(MaxPooling2D(3, 3))
model.add(Flatten())
# model.add(Conv2D(8, (2, 2)))

model.summary()