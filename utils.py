import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize
from vgg16 import vgg16
import math

def random_mini_batches(X, Y, mini_batch_size):
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def test(X_test, Y_test, sess):
        X = tf.placeholder(dtype=tf.float32, shape=[None, 224,224,3])
        Y = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        vgg = vgg16(X, 'init_weights.npz', sess)
        predict_op = tf.argmax(vgg.fc4l, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # train_accuracy = accuracy.eval({vgg.imgs: X_train, Y: Y_train})
        test_accuracy = sess.run(accuracy, feed_dict={vgg.imgs: X_test, Y: Y_test})
        # print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
   