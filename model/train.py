import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import copy
import math
import os

#mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

path = '../NR-ER/\
NR-ER-train/names_onehots.npy'
labels = np.genfromtxt('../NR-ER/\
NR-ER-train/names_labels.csv',delimiter=',',dtype=None)
data = np.load(path, allow_pickle = True)
m = np.atleast_1d(data)
dictionary = dict(m[0])
onehotArr = dictionary['onehots']

onehotArr = onehotArr.astype('float32')
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
        label[m] = [1.0, 0.0]
    else:
        label[m] = [0.0, 1.0]
#print(label)


'''CONFIGS...'''
pointer = 0
batchSize = 64
numOfUnroll = onelength
vocabulary = dictlength
trainstep = 100000
learningRate = 0.001
'''END OF CONFIG'''
dense_num = 256
second_dence = 256
n = 2

xs = tf.placeholder(tf.float32, [None, vocabulary, onelength], name = 'x_input')
ys = tf.placeholder(tf.float32, [None, n], name = 'y_input')

weights = {
    'in': tf.Variable(tf.random_normal([onelength, dense_num]), name = 'inW'),
    'middle': tf.Variable(tf.random_normal([dense_num, second_dence]), name = 'midW'),#72*1024 = 73728
    'out': tf.Variable(tf.random_normal([second_dence, n]), name = 'outW')
}
biases = {
    'in': tf.Variable(tf.constant(0., shape = [dense_num, ]), name = 'inB'),
    'middle': tf.Variable(tf.constant(0., shape = [second_dence, ]), name = 'midB'),
    'out': tf.Variable(tf.constant(0., shape = [n, ]), name = 'outB')
}

def RNN(X, weight, biases):
    X = tf.reshape(X, [-1, onelength])
    In = tf.matmul(X, weights['in']) + biases['in']
    In = tf.reshape(In, [-1, vocabulary, dense_num])
    cell = tf.contrib.rnn.BasicLSTMCell(dense_num, forget_bias = 1.0, state_is_tuple = True)
    init_state = cell.zero_state(batchSize, dtype = tf.float32)
    outputs, finalState = tf.nn.dynamic_rnn(cell, In, initial_state = init_state, time_major = False)
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    #result = tf.matmul(outputs[-1], weight['middle']) + biases['middle']
    #result = tf.nn.relu(result)
    results = (tf.matmul(outputs[-1], weights['out']) + biases['out'])
    return results

pred = RNN(xs, weights, biases)
cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred, labels = ys)
trainOp = tf.train.AdamOptimizer(learningRate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def getBA(pred, ys):
    for m in range(0, len(pred)):
        k = 0
init = tf.global_variables_initializer()
  #//to point the batch


def getnextBatch(data, label, batch_size):
    index = [i for i in range(0, datalength)]
    count = 0
    batch_xs = []
    batch_ys = []
    np.random.shuffle(index)
    for m in range(0, datalength):
        if label[index[m]][1] == 1:
            batch_xs.append(data[index[m]])
            batch_ys.append(label[index[m]])
            count+=1
        elif m%6 == 0:
            batch_xs.append(data[index[m]])
            batch_ys.append(label[index[m]])
            count+=1
        if count==batch_size:
            break
    return batch_xs, batch_ys

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step*batchSize < trainstep:
        #batch_xs, batch_ys = getnextBatch(onehotArr, label, batchSize)
        #batch_xs, batch_ys = mnist.train.next_batch(batchSize)
        #batch_xs = get10(batch_xs)
        #if step%100==0:
            #print(batch_xs[0])
        #batch_xs = batch_xs.reshape([batchSize, vocabulary, onelength])
        #print('NOOOO: ', always0(batch_ys))
        #print(sess.run(cost, feed_dict = {xs: batch_xs, ys: batch_ys}))
        sess.run([trainOp], feed_dict = {xs: onehotArr[0:64], ys: label[0:64]})
        if step%1 ==0:
            #print(batch_xs[0], batch_ys, cost)
            print(sess.run(pred, feed_dict = {xs: onehotArr[0:64], ys: label[0:64]}))
            print(sess.run(accuracy, feed_dict = {xs: onehotArr[0:64], ys: label[0:64]}))
        step = step + 1
    savePath = saver.save(sess, '../bin/RNNmodel256.ckpt')
    print(savePath)
## 0.1: learning rate

'''


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
'''
