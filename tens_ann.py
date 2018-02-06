import tensorflow as tf
import numpy
from parser_enc import *
from functions import *

x = tf.placeholder(tf.float32, shape = [None, 22])
y_ = tf.placeholder(tf.float32, shape = [None, 2])

W_1 = weight_initializer([22,100])
b_1 = bias_initializer([100])

h_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)

W_2 = weight_initializer([100,2])
b_2 = bias_initializer([2])

y_out = tf.matmul(h_1, W_2) + b_2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_out))

train_step = tf.train.AdamOptimizer(1e-1).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_out,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

inputs, labels = input_inject_GLIOMA_MLP()
bsize=5
epochs = 1000
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for j in xrange(epochs):

        print "Epoch: " + str(j)

        for i in xrange(10):

            if i%1 == 0:

                train_accuracy, error = sess.run([accuracy, cross_entropy], feed_dict={x:inputs[bsize*(i):bsize*(i+1)], y_:labels[bsize*(i):bsize*(i+1)]})

                print('Step %d, Training accuracy %g Error %f \n' % (i, train_accuracy, error))


            train_step.run(feed_dict={x:inputs[bsize*(i):bsize*(i+1)], y_:labels[bsize*(i):bsize*(i+1)]})





    print "\nTest\n"
    test_accuracy, test_error = sess.run([accuracy, cross_entropy], feed_dict={x:inputs[50:], y_:labels[50:]})
    print "Accuracy = " + str(test_accuracy)
    print "Error = " + str(test_error)
