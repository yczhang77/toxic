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
numOfUnroll = onelength+2
vocabulary = dictlength
trainstep = 100
learningRate = 0.0001
'''END OF CONFIG'''
dense_num = 256
second_dence = 256
n = 2

def paddingzero(data):
    for m in range(0, len(data)):
        for n in range(0, 72):
            np.append(data[m][n], 0)
            np.append(data[m][n], 0)
    return data

xs = tf.placeholder(tf.float32, [None, vocabulary, onelength], name = 'x_input')
ys = tf.placeholder(tf.float32, [None, n], name = 'y_input')
NDprob = tf.placeholder(tf.float32)
onedrug = tf.reshape(xs, [-1, vocabulary, onelength, 1])
def weight_init(shape, name):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial, name = name)
def bias_init(shape, name):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial, name = name)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = "SAME")
def maxpooling(x, mul1, mul2):
    return tf.nn.max_pool(x, ksize = [1, mul1, mul1, 1], strides = [1, mul1, mul1, 1], padding = "SAME")

Wcov1 = weight_init([5, 5, 1, 32], "wconv1")
Bcov1 = bias_init([32], "bconv1")

Wcov2 = weight_init([5, 5, 32, 64], "wconv2")
Bcov2 = bias_init([64], "bconv2")

wfc1 = weight_init([18*100*64, 1024], "wconv3")
bfc1 = bias_init([1024], "bconv3")

wfc2 = weight_init([1024, n], "wconv4")
bfc2 = bias_init([n], "bconv4") 

conv1 = tf.nn.relu(conv2d(onedrug, Wcov1)+ Bcov1)
pool1 = maxpooling(conv1, 2, 2)
conv2 = tf.nn.relu(conv2d(pool1, Wcov2)+ Bcov2)
pool2 = maxpooling(conv2, 2, 2)
pool2flat = tf.reshape(pool2, [-1, 18*100*64])
fcl1 = tf.nn.relu(tf.matmul(pool2flat, wfc1) + bfc1)
fc1do = tf.nn.dropout(fcl1, NDprob)
output = tf.matmul(fc1do, wfc2) + bfc2

cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits = output, labels = ys)
trainOp = tf.train.AdamOptimizer(learningRate).minimize(cost)

correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def getBA(pred, ys):
    for m in range(0, len(pred)):
        k = 0
init = tf.global_variables_initializer()
  #//to point the batch


def getnextBatch(data, label, batch_size):
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
    data = paddingzero(onehotArr)
    while step*batchSize < trainstep:
        batch_xs, batch_ys = getnextBatch(data, label, batchSize)
        #batch_xs, batch_ys = mnist.train.next_batch(batchSize)
        #batch_xs = get10(batch_xs)
        #if step%100==0:
            #print(batch_xs[0])
        #batch_xs = batch_xs.reshape([batchSize, vocabulary, onelength])
        #print('NOOOO: ', always0(batch_ys))
        #print(sess.run(cost, feed_dict = {xs: batch_xs, ys: batch_ys}))
        sess.run([trainOp], feed_dict = {xs: batch_xs, ys: batch_ys, NDprob: 0.5})
        if step%10 ==0:
            #print(batch_xs[0], batch_ys, cost)
            
	    #print(sess.run(pred, feed_dict = {xs: onehotArr[0:64], ys: label[0:64]}))
            print(sess.run(output, feed_dict = {xs:data[0:100], ys:label[0:100], NDprob: 1.0}))
            print(sess.run(accuracy, feed_dict = {xs: data[0:100], ys: label[0:100], NDprob: 1.0}))
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
