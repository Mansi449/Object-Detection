import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize
from vgg16 import vgg16
import math
from utils import *

def model(X_train, Y_train, learning_rate=0.009, X=None, Y=None, weights=None, sess=None, num_epochs=3, minibatch_size=5):
    m = X_train.shape[0]
    if weights is not None and sess is not None:
        vgg = vgg16(X, weights, sess)
    fc4l = vgg.fc4l
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc4l, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(num_epochs):
        minibatch_cost = 0.
        num_minibatches = int(m / minibatch_size)
        minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
        for (minibatch_X, minibatch_Y) in minibatches:
            _, temp_cost = sess.run([optimizer, cost], feed_dict={Y:minibatch_Y, vgg.imgs:minibatch_X})
            print(temp_cost)

def test(X_test, Y_test, sess):
        X = tf.placeholder(dtype=tf.float32, shape= [None, 224,224,3])
        vgg = vgg16(X, 'init_weights.npz', sess)
        predict_op = tf.argmax(vgg.fc4l, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # train_accuracy = accuracy.eval({vgg.imgs: X_train, Y: Y_train})
        test_accuracy = sess.run(accuracy, feed_dict={vgg.imgs: X_test, Y: Y_test})
        # print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
                

sess = tf.Session()
X = tf.placeholder(tf.float32, [None, 224, 224, 3])
Y = tf.placeholder(tf.float32, [None, 2])
# Y = tf.constant([0,1], dtype = tf.float32)
X_train = np.random.randint(225,size=(11,224,224,3))
Y_train = np.random.randint(10,size=(11,2))
count = -1
for y in Y_train:
    count += 1
    Y_train[count] = [0,1] if Y_train[count][0]<=Y_train[count][1] else [1,0]
img1 = imread('img.png', mode='RGB')
img1 = imresize(img1, (224, 224))
test(X_train, Y_train, sess)
# model(X_train, Y_train, 0.009, X, Y, 'init_weights.npz', sess)