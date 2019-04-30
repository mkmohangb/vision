from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Flatten, Input, MaxPooling2D
import matplotlib.pyplot as plt

batch_size = 32
num_classes = 10
epochs = 50
num_predictions = 20

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def conv_block(input_tensor, num_channels, padding, batch_norm):
    x = Conv2D(num_channels, (3, 3), padding=padding)(input_tensor)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

# Architecture taken from Keras CIFAR-10 CNN tutorial (Dropout removed)
def get_model(batch_norm):
    input = Input(x_train.shape[1:])
    x = conv_block(input, 32, 'same', batch_norm)
    x = conv_block(x, 32, 'valid', batch_norm)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = conv_block(x,     64, 'same', batch_norm)
    x = conv_block(x, 64, 'valid', batch_norm)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Flatten()(x)
    x = Dense(512)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(num_classes)(x)
    output = Activation('softmax')(x)

    model = Model(input, output)
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    return model

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

history = get_model(False).fit(x_train, y_train, batch_size=batch_size, epochs=epochs,validation_data=(x_test, y_test))
history_bn = get_model(True).fit(x_train, y_train, batch_size=batch_size, epochs=epochs,validation_data=(x_test, y_test))

plt.plot(history.history['val_acc'])
plt.plot(history_bn.history['val_acc'])
plt.title('Model accuracy: CIFAR-10')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Without BN', 'With BN'], loc='lower right')
plt.savefig('bn_expt.png')
plt.show()
