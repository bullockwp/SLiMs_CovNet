import deepfold_batch_factory as dbf
import numpy as np
import glob
import os
import sys

import tensorflow as tf

FLAGS = None

##
# # # Building a conv-NN model

# weight & bias initializations
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# convolution and pooling functions

def conv3d(x, W):
    ''' 3d convolution'''
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

def max_pool(x):
    '''chooses the largest abs value in pool'''
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

# defining the model itself

def conv_NN(x):
    """
    Learns stuff

    :param x: A 5D tensor of variabels
    :return:  A 2D tensor of labels
    """

    # reshape data?!?!
    x_image = x

    # Layer 1
    # in 2 / out 32
    W_conv1 = weight_variable([5, 5, 5, 2, 16])
    b_conv1 = bias_variable([16])


    h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1)
    print h_pool1.shape

    # Layer 2
    # in 32 / out 64
    W_conv2 = weight_variable([5, 5, 5, 16, 16])
    b_conv2 = bias_variable([16])


    h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2)
    print h_pool2.shape

    # Densely connected layer
    # in 64 / out 1024
    W_fc1 = weight_variable([6 * 19 * 38 * 16, 32])
    # each parameter is the related input tensor divided by the Ksize set for max_pool, each time you applied max pool, multiplied by the input channels

    b_fc1 = bias_variable([32])

    #Workaround- flattening
    dim = np.prod(h_pool2.get_shape().as_list()[1:])
    h_pool2_flat = tf.reshape(h_pool2, [-1, dim])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout goes here later...
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout

    W_fc2 = weight_variable([32, 21])
    b_fc2 = bias_variable([21])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, keep_prob


##
# # # Training the model

if __name__ == '__main__':
    # #Preliminaries
    ##define initial variable parameters for batch extraction

    input_dir = "./atomistic_features_spherical"
    test_set_fraction = 0.25
    validation_set_size = 0.10
    max_batch_size = 2
    num_passes = 2

    ##Preliminary data invocation

    # Read in names of all feature files
    protein_feature_filenames = sorted(glob.glob(os.path.join(input_dir, "*protein_features.npz")))
    grid_feature_filenames = sorted(glob.glob(os.path.join(input_dir, "*residue_features.npz")))

    # Set range for validation and test set
    validation_end = test_start = int(len(protein_feature_filenames) * (1. - test_set_fraction))
    train_end = validation_start = int(validation_end - validation_set_size)

    # Create batch factory and add data sets

    #Training set
    batch_factory = dbf.BatchFactory()
    batch_factory.add_data_set("data", protein_feature_filenames[:train_end], grid_feature_filenames[:train_end])
    batch_factory.add_data_set("model_output", protein_feature_filenames[:train_end], key_filter=["aa_one_hot"])

    #Validation set
    validation_batch_factory = dbf.BatchFactory()
    batch_factory.add_data_set("data", protein_feature_filenames[validation_start:validation_end], grid_feature_filenames[validation_start:validation_end])
    batch_factory.add_data_set("model_output", protein_feature_filenames[validation_start:validation_end], key_filter=["aa_one_hot"])

    total_data_size = batch_factory.data_size()

    x = tf.placeholder(tf.float32, shape=[None, 24, 76, 151, 2])
    y_ = tf.placeholder(tf.float32, shape=[None, 21])

    y_conv, keep_prob = conv_NN(x)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

    #this is what is evaluated below
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print "Training started!"

        total_correct = 0.
        total_all = 0.

        for i in range(num_passes):
            print "pass ", i

            data_size = 0
            print "Data Size is ", data_size

            while data_size < total_data_size:
                batch, _ = batch_factory.next(max_batch_size)

                grid_matrix = batch["data"]
                labels = batch["model_output"]

                data_size += labels.shape[0]

                if data_size % 1 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x: grid_matrix, y_: labels, keep_prob: 1.0})
                    print('Data Size is %d, training accuracy %g' % (data_size, train_accuracy))
                    print "correct / all", total_correct, "/", total_all
                    total_correct = total_correct + train_accuracy
                    total_all = total_all + max_batch_size
                    # Total_Training_accuracy = (total_correct / total_all)
                    # print "the total training accuracy was", Total_Training_accuracy
                train_step.run(feed_dict={x: grid_matrix, y_: labels, keep_prob: 0.5})


        Total_Training_accuracy = (total_correct / total_all)
        print "the total training accuracy was", Total_Training_accuracy

        print "Validation started!"
        #
        #
        # data_size = 0
        # print "Data Size is ", data_size
        #
        # while data_size < total_data_size * validation_set_size:
        #     batch, _ = validation_batch_factory.next(max_batch_size)
        #
        #     grid_matrix = batch["data"]
        #     labels = batch["model_output"]
        #
        #     data_size += labels.shape[0]




