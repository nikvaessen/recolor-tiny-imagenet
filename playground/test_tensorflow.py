import numpy as np
import keras.backend as K

import tensorflow as tf


def test():
    # v = K.argmax(y_true, axis=3)
    # v = K.one_hot(v, 262)
    #
    # weights = K.variable(weight_list)
    # test = weights * v
    # print('Test', test.shape)
    # weights = K.expand_dims(weights, axis=0)
    # weights = K.repeat_elements(weights, rep=64, axis=0)
    # print('update')
    #
    # print('update')
    # print(weights.shape)
    # v = K.dot(v, weights)

    ############################""
    a = np.array([[[5, 2, 6],
                    [7, 7, 3],
                    [0, 0, 0],
                    [6, 6, 5]],

                   [[2, 2, 3],
                    [1, 3, 3],
                    [8, 1, 9],
                    [0, 5, 7]]])
    weights = np.array([2, 0, 1])
    a = K.constant(a)
    b = K.argmax(a, axis=2)
    c = K.one_hot(b, 3)
    d = c * weights
    e = K.sum(d, axis=2)
    sess = tf.Session()
    print(sess.run(a))
    print('*_' * 10)
    print(sess.run(b))
    print('*_' * 10)
    print(sess.run(c))
    print('*_' * 10)
    print(sess.run(d))
    print('*_' * 10)
    print(sess.run(e))


test()
