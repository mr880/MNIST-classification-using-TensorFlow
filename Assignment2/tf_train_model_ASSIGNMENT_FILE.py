import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')



### suggests convolutions layers for 97%!!!!!!!!!!!
#### use math plot lib to plot all the information
class build_train:
    def __init__(self):
        self.rootPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir))   # DO NOT EDIT
        self.save_dir = self.rootPath + '/tf_model'   # DO NOT EDIT



    def build_train_network(self, network):


        ############### MNIST DATA #########################################
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)      # DO NOT EDIT
        ############### END OF MNIST DATA ##################################

        ############### CONSTRUCT NEURAL NETWORK MODEL HERE ################


        x = tf.placeholder(tf.float32, [None, 784], name='ph_x')
        y_ = tf.placeholder(tf.float32, [None, 10], name='ph_y_')

        W1 = tf.Variable(tf.zeros([784,10]))
        b1 = tf.Variable(tf.zeros([10]))


        #first convolusion layer

        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        x_image = tf.reshape(x, [-1,28,28,1])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)


        #second convolusion layer
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        #densely connected layer
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        #dropout
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        #readout layer
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2





        ## if you want extra credit u need to expand this to have a w2, b2 etc (extra layers)
        # OUTPUT VECTOR y MUST BE LENGTH 10, EACH OUTPUT NEURON CORRESPONDS TO A DIGIT 0-9
        ####the line below is the model
        y = tf.nn.softmax(tf.matmul(x, W1) + b1, name='op_y')



        ############# END OF TRAINING FUNCTION #############################
        cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='op_accuracy')


        ############# CONSTRUCT TRAINING SESSION ###########################
        saver = tf.train.Saver()                                            # DO NOT EDIT
        sess = tf.InteractiveSession()                                      # DO NOT EDIT
        sess.run(tf.global_variables_initializer())                         # DO NOT EDIT

        train_eval = []
        val_eval = []
        test_eval = []
        time_eval = []



        for i in range(5000):
            batch1 = mnist.train.next_batch(50)
            batch2 = mnist.validation.next_batch(50)
            batch3 = mnist.test.next_batch(50)

            if i%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x:batch1[0], y_: batch1[1], keep_prob: 1.0})
                print("step %d, training accuracy %g"%(i, train_accuracy))
                train_eval.append(train_accuracy)
            train_step.run(feed_dict={x: batch1[0], y_: batch1[1], keep_prob: 0.5})
            if i%100 == 0:
                validation_accuracy = accuracy.eval(feed_dict={
                    x:batch2[0], y_: batch2[1], keep_prob: 1.0})
                print("step %d, validation accuracy %g"%(i, validation_accuracy))
                val_eval.append(validation_accuracy)
            if i%100 == 0:
                test_accuracy = accuracy.eval(feed_dict={
                    x:batch3[0], y_: batch3[1], keep_prob: 1.0})
                print("step %d, test accuracy %g"%(i, test_accuracy))
                test_eval.append(test_accuracy)

                time_eval.append(i)  #this may need to change to time.append(i)

        print("test accuracy %g"%accuracy.eval(feed_dict= {
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
        


#### can switch to other data sets for your report to report different things.... (commented out stuff) 

        ############# END OF TRAINING SESSION ##############################

        ############# SAVE MODEL ###########################################

        saver.save(sess, save_path=self.save_dir, global_step=network)      # DO NOT EDIT
        print('Model Saved')                                                # DO NOT EDIT
        sess.close()                                                        # DO NOT EDIT
        ############# END OF SAVE MODEL ####################################

        ############# OUTPUT ACCURACY PLOT ################################


        plt.plot(time_eval, train_eval, 'b', time_eval, val_eval, 'r', time_eval, test_eval, 'g')
        plt.xlabel('iterations')
        plt.ylabel('accuracy')
        plt.title('training accuracy evaluation')
        plt.show()

        ############# END OF ACCURACY PLOT ################################
