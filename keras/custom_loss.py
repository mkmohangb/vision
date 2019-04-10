from keras.layers import Dense, Input
from keras.models import Model
from keras import backend as K
import numpy as np


def custom_loss(layer):
    print_op = K.tf.print('layer_weights:', layer, summarize=-1)
    def binary_crossentropy(y_true, y_pred):
        l2 = 0.001
        y_true = K.print_tensor(y_true, 'y_true: ')
        print_y_pred = K.tf.print({'y_pred': y_pred}, summarize=-1)
        with K.tf.control_dependencies([print_op, print_y_pred]):
            loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1) + K.sum(l2 * K.square(layer))
            loss = K.tf.Print(loss, [loss], 'loss is: ', summarize=-1)
        return loss

    return binary_crossentropy


input = Input((10,))
hidden = Dense(5)(input)
output = Dense(1, activation='sigmoid')(hidden)
model = Model(input, output)

model.summary()
model.compile(optimizer='sgd', loss=custom_loss(model.layers[1].weights[0]), metrics=['accuracy'])

model.fit(np.random.random((2,10)), np.zeros((2,1)), batch_size=1, epochs=2)


