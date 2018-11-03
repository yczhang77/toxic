import tensorflow as tf
import numpy as np
import copy
import sys
import os

path = '../NR-ER/\
NR-ER-test/names_onehots.npy'
labels = np.genfromtxt('../NR-ER/\
NR-ER-test/names_labels.csv',delimiter=',',dtype=None)
data = np.load(path, allow_pickle = True)
m = np.atleast_1d(data)
dictionary = dict(m[0])
onehotArr = dictionary['onehots']
nameArr = dictionary['names']
dictlength = len(onehotArr[0]) #72， inputs
datalength = len(onehotArr)
onelength = len(onehotArr[0][0]) #300， time

label = []
for m in range(0, datalength):
    label.append([])
    label[m].append(labels[m][1])
    #print(label[m])
    if label[m] == [0]:
        #print(000)
        label[m] = [1, 0]
    else:
        label[m] = [0, 1]

pointer = 0
batchSize = len(label)
numOfUnroll = onelength
vocabulary = dictlength
trainstep = datalength
learningRate = 0.001
'''END OF CONFIG'''
dense_num = 256

weights = {
    'in': tf.get_variable('inW', shape = [dictlength, dense_num]),
    'out': tf.get_variable('outW', shape = [dense_num, 2])
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape = [dense_num, ]), name = 'inB'),
    'out': tf.Variable(tf.constant(0.5, shape = [2, ]), name = 'outB')
}
xs = tf.placeholder(tf.float32, [None, vocabulary, onelength], name = 'x_input')
ys = tf.placeholder(tf.float32, [None, 2], name = 'y_input')


def RNN(X, weight, biases):
    X = tf.reshape(X, [-1, dictlength])
    In = tf.matmul(X, weights['in']) + biases['in']

    In = tf.reshape(In, [-1, onelength, dense_num])

    cell = tf.contrib.rnn.BasicLSTMCell(dense_num)
    init_state = cell.zero_state(batchSize, dtype = tf.float32)
    outputs, finalState = tf.nn.dynamic_rnn(cell, In, initial_state = init_state, time_major = False)
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return results

pointer = 0
def getnextBatch(data, label, batch_size):
    global pointer
    Range = np.arange(batch_size)
    np.random.shuffle(Range)
    if  batch_size + pointer <= datalength - 1:
        batch_xS = data[pointer:pointer + batch_size]
        #print(len(batch_xs[0][0]), len(batch_xs))
        batch_yS = label[pointer:pointer + batch_size]
        batch_xs = np.zeros((batch_size, len(data[0]), len(data[0][0])))
        batch_ys = np.zeros((batch_size, len(label[0])))
        for count in range(0, batchSize):
            batch_xs[count] = batch_xS[Range[count]]
            batch_ys[count] = batch_yS[Range[count]]
        pointer = pointer + batch_size
    else:
        batch_x = data[pointer: datalength]
        batch_xS = np.vstack((batch_x, data[0 : (batch_size - (datalength% batch_size))]))
        #print(batch_xs, len(batch_xs), len(batch_x),  (batch_size - (datalength)% batch_size), pointer%batch_size, pointer)
        batch_y = label[pointer: datalength]
        print(len(batch_y),  len(batch_y[0]))

        batch_yS = np.vstack((batch_y, label[0 : (batch_size  - (datalength % batch_size))]))
        batch_xs = np.zeros((batch_size, len(data[0]), len(data[0][0])))
        batch_ys = np.zeros((batch_size, len(label[0])))
        for count in range(0, batchSize):
            batch_xs[count] = batch_xS[Range[count]]
            batch_ys[count] = batch_yS[Range[count]]
        pointer = datalength % batch_size
    return batch_xs, batch_ys

pred = RNN(xs, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred, labels = ys))
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, '../bin/RNNmodel256.ckpt')
    sess.run(init)
    step = 0
    while step*batchSize < trainstep:
        batch_xs, batch_ys = getnextBatch(onehotArr, label, batchSize)
        batch_xs = batch_xs.reshape([batchSize, vocabulary, onelength])
        print(sess.run(accuracy, feed_dict = {xs: batch_xs, ys: batch_ys}))
        #print(sess.run(cost, feed_dict = {xs: batch_xs, ys: batch_ys}))
        #sess.run([trainOp], feed_dict = {xs: batch_xs, ys: batch_ys})
        step = step + 1
    #savePath = saver.save(sess, '../RNNmodel.ckpt')
    #print(savePath)
