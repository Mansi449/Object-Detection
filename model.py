import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize
from vgg16 import vgg16
import math
import matplotlib.pyplot as plt
from utils import *
import h5py

def model(X_train, Y_train, learning_rate=0.009, X=None, Y=None, weights=None, sess=None, num_epochs=10, minibatch_size=5, print_cost=True):
    m = X_train.shape[0]
    costs = []
    if weights is not None and sess is not None:
        vgg = vgg16(X, weights, sess)
    fc3l = vgg.fc3l
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc3l, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(num_epochs):
        minibatch_cost = 0.
        num_minibatches = int(m / minibatch_size)
        minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
        for (minibatch_X, minibatch_Y) in minibatches:
            _, temp_cost = sess.run([optimizer, cost], feed_dict={Y:minibatch_Y, vgg.imgs:minibatch_X})
            minibatch_cost += temp_cost / num_minibatches
        # Print the cost every epoch
        if print_cost == True and epoch % 1 == 0:
            print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
        if print_cost == True and epoch % 1 == 0:
            costs.append(minibatch_cost)
    params = {}
    keys = []
    for key in sorted(np.load(weights)):
        keys.append(key)
    index = -1

    for item in vgg.parameters:
        index += 1
        params[keys[index]] = sess.run(item)
    np.savez("output_weight", **params)
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

sess = tf.Session()
face_count = 0
non_face_count = 5
file1 = h5py.File('faceData_v2.h5','r')
file2 = h5py.File('nonface_small.h5','r')
Y = np.array([[1,0]])
X_train1 = np.array(file1["X_train"])
X_train2 = np.array(file2["X_train_30000"])
count = -1
fac = np.array([[1,0]])
non_fac = np.array([[0,1]])
for data in X_train1:
    count += 1
    if count == face_count:
        break
    Y = np.concatenate((Y,fac),axis=0)
for data in X_train2:
    if count == non_face_count+ face_count:
        break
    count += 1
    Y = np.concatenate((Y,non_fac),axis=0)
X = np.concatenate((X_train1[0:face_count,:,:,:],X_train2[0:non_face_count,:,:,:]), axis=0)
X_train1 = None
X_train2 = None
permutation = np.random.permutation(count)
X_train = X[permutation,:,:,:]
Y_train = Y[permutation,:]
X = tf.placeholder(tf.float32, [None, 224, 224, 3])
Y = tf.placeholder(tf.float32, [None, 2])
print('data loaded')
test(X_train, Y_train, sess)
# model(X_train, Y_train, 0.009, X, Y, 'output_weight.npz', sess)