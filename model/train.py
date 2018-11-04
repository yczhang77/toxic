import tensorflow as tf
import numpy as np
import copy
import math
import os

path = '../NR-ER/\
NR-ER-train/names_onehots.npy'
labels = np.genfromtxt('../NR-ER/\
NR-ER-train/names_labels.csv',delimiter=',',dtype=None)
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


'''CONFIGS...'''
pointer = 0
batchSize = 64
numOfUnroll = onelength
vocabulary = dictlength
trainstep = datalength*5
learningRate = 0.01
'''END OF CONFIG'''
dense_num = 256
second_dence = 256

xs = tf.placeholder(tf.float32, [None, vocabulary, onelength], name = 'x_input')
ys = tf.placeholder(tf.float32, [None, 2], name = 'y_input')

weights = {
    'in': tf.Variable(tf.random_normal([dictlength, dense_num]), name = 'inW'),
    'middle': tf.Variable(tf.random_normal([dense_num, second_dence]), name = 'midW'),#72*1024 = 73728
    'out': tf.Variable(tf.random_normal([second_dence, 2]), name = 'outW')
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape = [dense_num, ]), name = 'inB'),
    'middle': tf.Variable(tf.constant(0.1, shape = [second_dence, ]), name = 'midB'),
    'out': tf.Variable(tf.constant(0.1, shape = [2, ]), name = 'outB')
}

def RNN(X, weight, biases):
    X = tf.reshape(X, [-1, dictlength])
    In = tf.matmul(X, weights['in']) + biases['in']

    In = tf.reshape(In, [-1, onelength, dense_num])

    cell = tf.contrib.rnn.BasicLSTMCell(dense_num)
    init_state = cell.zero_state(batchSize, dtype = tf.float32)
    outputs, finalState = tf.nn.dynamic_rnn(cell, In, initial_state = init_state, time_major = False)
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    #result = tf.matmul(outputs[-1], weight['middle']) + biases['middle']
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return results

pred = RNN(xs, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred, labels = ys))
trainOp = tf.train.AdamOptimizer(learningRate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
  #//to point the batch

def getnextBatch(data, label, batch_size):
<<<<<<< HEAD
    global pointer
    Range = np.arange(batch_size)
    np.random.shuffle(Range)
=======
    index = [i for i in range(batch_size, datalength)]
    count = 0
    batch_xs = []
    batch_ys = []
    np.random.shuffle(index)
    for m in range(0, datalength-batch_size):
        if label[index[m]][1] == 1:
            batch_xs.append(data[index[m]])
            batch_ys.append(label[index[m]])
            count+=1
        elif m%4 == 0:
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
    print(result)

with tf.Session() as sess:
    sess.run(init)
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
        sess.run([trainOp], feed_dict = {xs: batch_xs, ys: batch_ys})
        if step%100 ==0:
            #print(batch_xs[0], batch_ys, cost)
            
	    #print(sess.run(pred, feed_dict = {xs: onehotArr[0:64], ys: label[0:64]}))
            trueacc(sess.run(pred, feed_dict = {xs:onehotArr[0:64], ys:label[0:64]}), label[0:64])
            print(sess.run(accuracy, feed_dict = {xs: onehotArr[0:64], ys: label[0:64]}))
        step = step + 1
    savePath = saver.save(sess, '../bin/RNNmodel256.ckpt')
    print(savePath)
## 0.1: learning rate

'''


>>>>>>> 5cfbb1f02b6b039e8951ee55e752806ab6d124be
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

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step*batchSize < trainstep:
        batch_xs, batch_ys = getnextBatch(onehotArr, label, batchSize)
        batch_xs = batch_xs.reshape([batchSize, vocabulary, onelength])
        #print(sess.run(cost, feed_dict = {xs: batch_xs, ys: batch_ys}))
        sess.run([trainOp], feed_dict = {xs: batch_xs, ys: batch_ys})
        if step%1 ==0:
            print(sess.run(cost, feed_dict = {xs: batch_xs, ys: batch_ys}))
        step = step + 1
    savePath = saver.save(sess, '../bin/RNNmodel256.ckpt')
    print(savePath)
## 0.1: learning rate
