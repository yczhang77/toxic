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
    'in': tf.get_variable('inW', shape = [onelength, dense_num]),
    'out': tf.get_variable('outW', shape = [dense_num, 2])
}
biases = {
    'in': tf.Variable(tf.constant(0.0, shape = [dense_num, ]), name = 'inB'),
    'out': tf.Variable(tf.constant(0.0, shape = [2, ]), name = 'outB')
}
xs = tf.placeholder(tf.float32, [None, vocabulary, onelength], name = 'x_input')
ys = tf.placeholder(tf.float32, [None, 2], name = 'y_input')


#mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

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
    for m in range(0, batch_size):
        if label[index[m]][1] == 1:
            batch_xs.append(data[index[m]])
            batch_ys.append(label[index[m]])
            count+=1
        else:
            #m%4 == 0:
            batch_xs.append(data[index[m]])
            batch_ys.append(label[index[m]])
            count+=1
        if count==batch_size:
            break
    return batch_xs, batch_ys

saver = tf.train.Saver()

def trueacc(pred, label):
    a=0
    b=0
    c=0
    d=0
    for m in range(0, len(pred)):
        if pred[m][0]>pred[m][1]:
            if label[m][0] == 1:
                c+=1
            a+=1
        else:
            if label[m][1] ==1:
                d+=1
            b+=1
    result = 1/2*(c/(a+b-d)+d/(a+b-c))
    print('total 0 predicted: ', a)
    print('total 1 predicted: ', b)
    print('total 0 ok: ', c)
    print('total 1 ok: ', d)
    print(result)

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, '../bin/RNNmodel256.ckpt')
    step = 0
    while step*batchSize < trainstep:
        batch_xs, batch_ys = getnextBatch(onehotArr, label, batchSize)
        #batch_xs, batch_ys = mnist.train.next_batch(batchSize)
        #batch_xs = get10(batch_xs)
        #if step%100==0:
            #print(batch_xs[0])
        #batch_xs = batch_xs.reshape([batchSize, vocabulary, onelength])
        #print('NOOOO: ', always0(batch_ys))
        #print(sess.run(cost, feed_dict = {xs: batch_xs, ys: batch_ys}))
        sess.run([trainOp], feed_dict = {xs: onehotArr, ys: label})
        #if step%100 ==0:
            #print(batch_xs[0], batch_ys, cost)

	    #print(sess.run(pred, feed_dict = {xs: onehotArr[0:64], ys: label[0:64]}))
        trueacc(sess.run(pred, feed_dict =  {xs: onehotArr, ys: label}), label)
        print(sess.run(accuracy, feed_dict =  {xs: onehotArr, ys: label}))
        print(sess.run(pred, feed_dict =  {xs: onehotArr, ys: label}))
        step = step + 1
    #savePath = saver.save(sess, '../bin/RNNmodel256.ckpt')
    #print(savePath)
## 0.1: learning rate
