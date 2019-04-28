from keras.datasets import mnist
from keras.layers import Input, Dense, Flatten, Activation, BatchNormalization
from keras.models import Model
from keras.utils import to_categorical
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

## Experiment from the paper: Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

def dense_block(input_tensor, batch_norm):
    x = Dense(100, kernel_initializer='random_normal')(input_tensor)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    return x

def get_model(batch_norm):
    input = Input(shape=(28,28))
    x = Flatten()(input)
    x = dense_block(x, batch_norm)
    x = dense_block(x, batch_norm)
    x = dense_block(x, batch_norm)
    output = Dense(10, kernel_initializer='random_normal', activation='softmax')(x)
    model = Model(input, output)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
epochs = 50
history = get_model(False).fit(x_train, y_train, batch_size=60, epochs=epochs, validation_data=(x_test, y_test))
history_bn = get_model(True).fit(x_train, y_train, batch_size=60, epochs=epochs, validation_data=(x_test, y_test))

plt.plot(history.history['val_acc'])
plt.plot(history_bn.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Without BN', 'With BN'], loc='lower right')
plt.savefig('bn_expt.png')
plt.show()
