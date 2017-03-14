#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 05:34:56 2017

@author: vijay
"""

import pickle
import problem_unittests as tests
import helper
import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer

import utils

# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))



def neural_net_image_input(image_shape):
    """
    Return a Tensor for a bach of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    # TODO: Implement Function
    #print(image_shape)
    x = tf.placeholder(tf.float32,(None, image_shape[0], image_shape[1], image_shape[2]), name='x' )
    #print('Input Shape' , x.get_shape().as_list())

    tf.summary.image('input', x, 10)
    return x


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    # TODO: Implement Function
    #print(n_classes)
    y = tf.placeholder(tf.float32,(None, n_classes), name='y' )
    return y

def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    # TODO: Implement Function
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return keep_prob

def variable_summaries(var, scope):
    with tf.name_scope(scope):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.histogram('histogram', var)
        
def weight_variable(size, initializer, scope, weight_decay, show_kernels=False):
    
    weight = tf.get_variable('W', size, initializer=initializer)
    
    variable_summaries(weight, 'W')
    
    if show_kernels is True:
        grid = utils.put_kernels_on_grid(weight)
       # print('Kernel Image shape : ', grid.get_shape().as_list())
        tf.summary.image(scope, grid, max_outputs=3)
    if weight_decay is not 0.0:
        W_decay = tf.multiply(tf.nn.l2_loss(weight), weight_decay)
        tf.add_to_collection('losses', W_decay)
    return weight
    
def conv2d(x_tensor, conv_num_outputs, conv_ksize, conv_strides,
           padding='SAME',scope=None, activation='relu', initialize='he_normal',
           wd=0.0):
    
    in_channels = x_tensor.get_shape().as_list()[3]
    initializer = variance_scaling_initializer()
    if initialize == 'normal':
        initializer = tf.random_normal_initializer(stddev=5e-2)
        
    # filter size : [filter_height, filter_width, in_channels, out_channels]
    
    #weight = tf.Variable(tf.truncated_normal((conv_ksize[0], conv_ksize[1], in_channels, conv_num_outputs), stddev=5e-2))
    with tf.variable_scope(scope):
        
        weights = weight_variable([conv_ksize[0], conv_ksize[1], in_channels, conv_num_outputs],
                                 initializer, scope, wd, show_kernels=True)
    
        
        #bias = tf.Variable(tf.zeros(conv_num_outputs))
        bias = tf.get_variable('b',
                               [conv_num_outputs],
                               initializer=tf.constant_initializer(0.01))

        conv = tf.nn.conv2d(x_tensor, 
                            weights, 
                            strides=[1, conv_strides[0], conv_strides[1], 1],
                            padding=padding)
        conv = tf.nn.bias_add(conv, bias)
        
        if activation is 'relu':
            conv = tf.nn.relu(conv)
            
    return conv

def maxpool(x_tensor, pool_ksize, pool_strides, pool_type='max_pool'):
    
     if pool_type is 'frac_max_pool':
         out, _, _ = tf.nn.fractional_max_pool(x_tensor,
                                         pooling_ratio=[1., pool_ksize[0], pool_ksize[1], 1.],
                                         pseudo_random=True, overlapping=True)
     else:
         out = tf.nn.max_pool(x_tensor,
                              ksize = [1, pool_ksize[0], pool_ksize[1], 1],
                              strides = [1, pool_strides[0], pool_strides[1], 1],
                              padding = 'SAME')
     return out
 
def avgpool(x_tensor, pool_ksize, pool_strides):
     out = tf.nn.avg_pool(x_tensor,
                          ksize = [1, pool_ksize[0], pool_ksize[1], 1],
                          strides = [1, pool_strides[0], pool_strides[1], 1],
                          padding = 'SAME')
     return out

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides,
                   pool_ksize, pool_strides, padding='SAME', scope=None,
                   initialize='he_normal', wd=0.0, pool_type='max_pool'):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    # TODO: Implement Function
    initializer = variance_scaling_initializer()
    if initialize == 'normal':
        initializer = tf.random_normal_initializer(stddev=1.)
    
    # input size = [batch, in_height, in_width, in_channels]
    in_channels = x_tensor.get_shape().as_list()[3]
    
    # filter size : [filter_height, filter_width, in_channels, out_channels]
    #weight = tf.Variable(tf.truncated_normal((conv_ksize[0], conv_ksize[1], in_channels, conv_num_outputs), stddev=5e-2))

    with tf.variable_scope(scope):
        weights = weight_variable([conv_ksize[0], conv_ksize[1], in_channels, conv_num_outputs],
                                 initializer, scope, wd, show_kernels=True)
        #bias = tf.Variable(tf.zeros(conv_num_outputs))
        bias = tf.get_variable('b',
                               [conv_num_outputs],
                               initializer=tf.constant_initializer(0.00))
    
        conv = tf.nn.conv2d(x_tensor, 
                            weights, 
                            strides=[1, conv_strides[0], conv_strides[1], 1],
                            padding=padding)
        conv = tf.nn.bias_add(conv, bias)
        conv = tf.nn.relu(conv)
        
        if pool_type is 'frac_max_pool':
            conv_maxpool, _, _ = tf.nn.fractional_max_pool(x_tensor,
                                         pooling_ratio=[1., pool_ksize[0], pool_ksize[1], 1.],
                                         pseudo_random=True, overlapping=True)
        else:
            conv_maxpool = tf.nn.max_pool(conv,
                                          ksize = [1, pool_ksize[0], pool_ksize[1], 1],
                                          strides = [1, pool_strides[0], pool_strides[1], 1],
                                          padding = 'SAME')
    # print(conv_maxpool.get_shape().as_list())
    return conv_maxpool

def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    # TODO: Implement Function
    size = x_tensor.get_shape().as_list()
    x_tensor = tf.reshape(x_tensor, [-1, size[1]*size[2]*size[3]])
    return x_tensor

def fully_conn(x_tensor, num_outputs, scope=None, initialize='he_normal', wd=0.0):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    initializer = variance_scaling_initializer()
    if initialize == 'normal':
        initializer = tf.random_normal_initializer(stddev=5e-2)
        
    size = x_tensor.get_shape().as_list()
    with tf.variable_scope(scope):
        weights = weight_variable([size[1], num_outputs],
                                  initializer, scope, wd)
        bias = tf.get_variable('b',
                               [num_outputs],
                               initializer=tf.constant_initializer(0.01))
        
        #weights = tf.Variable(tf.truncated_normal((size[1], num_outputs), stddev=5e-2))
        #bias = tf.Variable(tf.zeros(num_outputs))
        dense = tf.add(tf.matmul(x_tensor, weights), bias)
        out = tf.nn.relu(dense)
    return out

def output(x_tensor, num_outputs, scope=None, initialize='he_normal', wd=0.0):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    initializer = variance_scaling_initializer()
    if initialize == 'normal':
        initializer = tf.random_normal_initializer(stddev=5e-2)

    size = x_tensor.get_shape().as_list()
    with tf.variable_scope(scope):
        weights = weight_variable([size[1], num_outputs],
                                  initializer, scope, wd)
        bias = tf.get_variable('b',
                               [num_outputs],
                               initializer=tf.constant_initializer(0.01))
        #weights = tf.Variable(tf.truncated_normal((size[1], num_outputs), stddev=5e-2))
        #bias = tf.Variable(tf.zeros(num_outputs))
        output = tf.add(tf.matmul(x_tensor, weights), bias)
    return output

def batch_norm(x, n_out, phase_train, scope=None):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def conv_net(x, keep_prob):
    conv = conv2d_maxpool(x, 32, (5, 5), (1, 1), (3, 3), (2, 2), scope='conv1',
                          initialize = 'normal')
    conv = tf.nn.dropout(conv, keep_prob)
    variable_summaries(conv, scope='act1_conv')
    flat = flatten(conv)
    dense = fully_conn(flat, 32, scope='dense1', initialize = 'normal')
    variable_summaries(conv, scope='act_dense')

    dense = tf.nn.dropout(dense, keep_prob)
    out = output(dense, 10, scope='dense2')
    variable_summaries(out, scope='act_out')
    return out
def conv_net_v0(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    """
    Currently gives a performance of around 76% for 100 epochs
    
    """
    
    # TODO: Apply 1, 2, or 3 Convolution and Max Pool layers
    #    Play around with different number of outputs, kernel size and stride
    # Function Definition from Above:
    #    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
    #drop = tf.nn.dropout(x, keep_prob)
    conv = conv2d_maxpool(x, 96, (5, 5), (1, 1), (3, 3), (2, 2), scope='conv1',
                          initialize = 'normal')
    conv = tf.nn.dropout(conv, keep_prob)
    #conv = conv2d_maxpool(conv, 96, (3, 3), (1, 1), (2, 2), (2, 2), scope='conv2',
    #                      initialize = 'normal')
    #conv = tf.nn.dropout(conv, keep_prob)
    #conv = conv2d_maxpool(conv, 128, (3, 3), (1, 1), (1, 1), (2, 2), scope='conv3',
    #                      initialize = 'normal')
    #conv = tf.nn.dropout(conv, keep_prob)
    
    # TODO: Apply a Flatten Layer
    # Function Definition from Above:
    #   flatten(x_tensor)
    
    flat = flatten(conv)

    # TODO: Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)
    
    dense = fully_conn(flat, 384, scope='dense1', initialize = 'normal')
    dense = tf.nn.dropout(dense, keep_prob)
    #dense = fully_conn(dense, 192, scope='dense2', initialize = 'normal')
    #dense = tf.nn.dropout(dense, keep_prob)
    
    # TODO: Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)
    
    out = output(dense, 10, scope='dense3')
    
    # TODO: return output
    return out

def conv_net_v1(x, keep_prob):
    
    #prob_1 = tf.minimum(keep_prob+0.35, 1.)
    #x = tf.nn.dropout(x, prob_1)
    
    # fractional max pooling
    """
    Currently gives a performance of around 78% for 100 epochs
    
    """
    
    
    conv = conv2d_maxpool(x, 96, (3, 3), (1,1), (1.44, 1.44), (2, 2), 
                          scope='conv1_1', wd=0.005, pool_type='frac_max_pool')
    conv = conv2d_maxpool(conv, 256, (3, 3), (1,1), (1.44, 1.44), (2, 2),
                          scope='conv1_2', wd=0.005, pool_type='frac_max_pool')
    conv = conv2d_maxpool(conv, 384, (3, 3), (1,1), (1.44, 1.44), (2, 2),
                          scope='conv1_3', wd=0.005, pool_type='frac_max_pool')
    conv = tf.nn.dropout(conv, keep_prob)
    
    conv = conv2d_maxpool(conv, 256, (3, 3), (1,1), (1.44,1.44), (2,2),
                          scope='conv2_1', wd=0.005, pool_type='frac_max_pool')
    conv = conv2d_maxpool(conv, 4096, (3, 3), (1,1), (1.44,1.44), (2,2),
                          scope='conv2_2', wd=0.005, pool_type='frac_max_pool')
    
    conv = tf.nn.dropout(conv, keep_prob)
   
    flat = flatten(conv)
    
    dense = fully_conn(flat, 512, scope='dense1', wd=0.005)
    dense = tf.nn.dropout(dense, keep_prob)
    dense = fully_conn(dense, 64, scope='dense2', wd=0.005)
    out = output(dense, 10, scope='dense3')
    
    return out

def conv_net_v2(x, keep_prob):
    
    # All convolutional net (No fully connected)
    # Accuracy of 77%
    prob_1 = tf.minimum(keep_prob+0.3, 1.)
    conv = tf.nn.dropout(x, prob_1)
    
    conv = conv2d(x, 96, (3, 3), (1,1), scope='conv1_1',
                  initialize='normal', wd=0.00)
    conv = conv2d(conv, 96, (3, 3), (1,1), scope='conv1_2',
                  initialize='normal', wd=0.00)
    conv = conv2d(conv, 96, (3, 3), (2,2), scope='conv1_3',
                  initialize='normal', wd=0.00)
    conv = tf.nn.dropout(conv, keep_prob)
    
    conv = conv2d(conv, 192, (3, 3), (1,1), scope='conv2_1',
                  initialize='normal', wd=0.0005)
    conv = conv2d(conv, 192, (3, 3), (1,1), scope='conv2_2',
                  initialize='normal', wd=0.0005)
    conv = conv2d(conv, 192, (3, 3), (2,2), scope='conv2_3',
                  initialize='normal', wd=0.0005)   
    conv = tf.nn.dropout(conv, keep_prob)
    
    conv = conv2d(conv, 192, (3, 3), (1,1), scope='conv3_1',
                  initialize='normal', wd=0.0005)
    conv = conv2d(conv, 192, (1, 1), (1,1), scope='conv3_2',
                  initialize='normal', wd=0.0005)
    conv = conv2d(conv, 10, (1, 1), (1,1), scope='conv3_3',
                  initialize='normal', wd=0.0005)

    conv = avgpool(conv, (8, 8), (8, 8))
    out = flatten(conv)
    #print('Shape of the Conv :', out.get_shape().as_list())
    
    return out
    
def conv_net_v3(x, keep_prob):
      
    #conv = conv2d(x, 64, (5, 5), (1,1))
    #maxp = maxpool(conv, (3, 3), (2, 2))
    #prob_1 = tf.minimum(keep_prob+0.5, 1.)
    #drop = tf.nn.dropout(x, prob_1)
    # with batch normalization
    """
    Currently gives a performance of around 83% for 300 epochs
    
    """
    phase_train = tf.less(keep_prob, 1.)
    
    conv = conv2d(x, 96, (5, 5), (1,1), scope='conv_1', activation='none')
    conv = batch_norm(conv,96, phase_train)
    conv = tf.nn.relu(conv)
    conv = maxpool(conv, (2, 2), (2, 2))
    
    conv = conv2d(conv, 256, (5, 5), (1,1), scope='conv_2', activation='none')
    conv = batch_norm(conv, 256, phase_train)
    conv = tf.nn.relu(conv)
    conv = maxpool(conv, (2, 2), (2, 2))
    
    conv = conv2d(conv, 384, (5, 5), (1,1), scope='conv_3', activation='none')
    conv = batch_norm(conv, 384, phase_train)
    conv = tf.nn.relu(conv)
    conv = maxpool(conv, (2, 2), (2, 2))
    
    conv = tf.nn.dropout(conv, keep_prob)
    
    conv = conv2d(conv, 256, (5, 5), (1,1), scope='conv_4', activation='none')
    conv = batch_norm(conv, 256, phase_train)
    conv = tf.nn.relu(conv)
    conv = maxpool(conv, (2, 2), (2, 2))
    
    conv = conv2d(conv, 4096, (5, 5), (1,1), scope='conv_5', activation='none')
    conv = batch_norm(conv, 4096, phase_train)
    conv = tf.nn.relu(conv)
    conv = maxpool(conv, (2, 2), (2, 2))
    
    flat = flatten(conv)
   
    dense = fully_conn(flat, 512, scope='dense1')
    dense = tf.nn.dropout(dense, keep_prob)
    dense = fully_conn(dense, 64, scope='dense2')
    out = output(dense, 10, scope='dense3')
    
    return out


def conv_net_v4(x, keep_prob):
    
    #Keep dropuout to 0 during training
    """
    Currently gives a performance of around 87% for 300 epochs
    
    """
    
    phase_train = tf.less(keep_prob, 1.)
    
    conv = conv2d(x, 96, (5,5), (1,1), scope='conv1_1', 
                  activation='none', wd=0.00)
    
    conv = batch_norm(conv, 96, phase_train, scope='bn1_1')
    conv = tf.nn.elu(conv)
    conv = tf.nn.dropout(conv, tf.minimum(keep_prob+0.3, 1.))
    conv = conv2d(conv, 96, (5,5), (1,1), scope='conv1_2',
                  activation='none', wd=0.00)
    conv = batch_norm(conv, 96, phase_train, scope='bn1_2')
    conv = tf.nn.elu(conv)
    print('Shape of the Conv :', conv.get_shape().as_list())
    #conv = maxpool(conv, (1.44, 1.44), (2, 2), pool_type='frac_max_pool')
    conv = maxpool(conv, (2, 2), (2, 2))
    print('Shape of the maxpool :', conv.get_shape().as_list())
    
    conv = conv2d(conv, 256, (3,3), (1,1), scope='conv2_1',
                  activation='none', wd=0.00)
    conv = batch_norm(conv, 256, phase_train, scope='bn2_1')
    conv = tf.nn.elu(conv)
    conv = tf.nn.dropout(conv, tf.minimum(keep_prob+0.4, 1.))
    conv = conv2d(conv, 256, (3,3), (1,1), scope='conv2_2',
                  activation='none', wd=0.0005)
    conv = batch_norm(conv, 256, phase_train, scope='bn2_2')
    conv = tf.nn.elu(conv)
    print('Shape of the Conv :', conv.get_shape().as_list())
    #conv = maxpool(conv, (1.44, 1.44), (2, 2), pool_type='frac_max_pool')
    conv = maxpool(conv, (2, 2), (2, 2))
    print('Shape of the maxpool :', conv.get_shape().as_list())
    
    conv = conv2d(conv, 384, (3,3), (1,1), scope='conv3_1', 
                  activation='none', wd=0.0005)
    conv = batch_norm(conv, 384, phase_train, scope='bn3_1')
    conv = tf.nn.elu(conv)
    conv = tf.nn.dropout(conv, tf.minimum(keep_prob+0.4, 1.))
    conv = conv2d(conv, 384, (3,3), (1,1), scope='conv3_2', 
                  activation='none', wd=0.0005)
    conv = batch_norm(conv, 384, phase_train, scope='bn3_2')
    conv = tf.nn.elu(conv)
    conv = tf.nn.dropout(conv, tf.minimum(keep_prob+0.4, 1.))
    conv = conv2d(conv, 384, (3,3), (1,1), scope='conv3_3', 
                  activation='none', wd=0.0005)
    conv = batch_norm(conv, 384, phase_train, scope='bn3_3')
    conv = tf.nn.elu(conv)
    print('Shape of the Conv :', conv.get_shape().as_list())
    #conv = maxpool(conv, (1.44, 1.44), (2, 2), pool_type='frac_max_pool')
    conv = maxpool(conv, (2, 2), (2, 2))
    print('Shape of the maxpool :', conv.get_shape().as_list())
    
    conv = conv2d(conv, 512, (3,3), (1,1), scope='conv4_1', 
                  activation='none', wd=0.0005)
    conv = batch_norm(conv, 512, phase_train, scope='bn4_1')
    conv = tf.nn.elu(conv)
    conv = tf.nn.dropout(conv, tf.minimum(keep_prob+0.4, 1.))
    conv = conv2d(conv, 512, (3,3), (1,1), scope='conv4_2', 
                  activation='none', wd=0.0005)
    conv = batch_norm(conv, 512, phase_train, scope='bn4_2')
    conv = tf.nn.elu(conv)
    conv = tf.nn.dropout(conv, tf.minimum(keep_prob+0.4, 1.))
    conv = conv2d(conv, 512, (3,3), (1,1), scope='conv4_3', 
                  activation='none', wd=0.0005)
    conv = batch_norm(conv, 512, phase_train, scope='bn4_4')
    conv = tf.nn.elu(conv)
    print('Shape of the Conv :', conv.get_shape().as_list())
    #conv = maxpool(conv, (1.44, 1.44), (2, 2), pool_type='frac_max_pool')
    conv = maxpool(conv, (2, 2), (2, 2))
    print('Shape of the maxpool :', conv.get_shape().as_list())
    
    conv = flatten(conv)
    conv = tf.nn.dropout(conv, tf.minimum(keep_prob+0.5, 1.))
    
    dense = fully_conn(conv, 1024, scope='dense1', wd=0.0005)
    dense = tf.nn.elu(dense)
    
    dense = tf.nn.dropout(dense, tf.minimum(keep_prob+0.5, 1.))
    dense = fully_conn(dense, 1024, scope='dense2', wd=0.0005)
    out = output(dense, 10, scope='dense3')
    #print(out.get_shape().as_list())
    return out

def conv_net_v5(x, keep_prob):
    
    #Keep dropuout to 0 during training
    """
    Currently gives a performance of around 87% for 300 epochs
    
    """
    
    phase_train = tf.less(keep_prob, 1.)
    
    conv = conv2d(x, 64, (3,3), (1,1), scope='conv1_1', 
                  activation='none', wd=0.00)
    conv = batch_norm(conv, 64, phase_train, scope='bn1_1')
    conv = tf.nn.elu(conv)
    conv = tf.nn.dropout(conv, tf.minimum(keep_prob+0.3, 1.))
    conv = conv2d(conv, 64, (3,3), (1,1), scope='conv1_2',
                  activation='none', wd=0.00)
    conv = batch_norm(conv, 64, phase_train, scope='bn1_2')
    conv = tf.nn.elu(conv)
    conv = maxpool(conv, (2, 2), (2, 2))
    
    conv = conv2d(conv, 128, (3,3), (1,1), scope='conv2_1',
                  activation='none', wd=0.0005)
    conv = batch_norm(conv, 128, phase_train, scope='bn2_1')
    conv = tf.nn.elu(conv)
    conv = tf.nn.dropout(conv, tf.minimum(keep_prob+0.4, 1.))
    conv = conv2d(conv, 128, (3,3), (1,1), scope='conv2_2',
                  activation='none', wd=0.0005)
    conv = batch_norm(conv, 128, phase_train, scope='bn2_2')
    conv = tf.nn.elu(conv)
    conv = maxpool(conv, (2, 2), (2, 2))
    
    conv = conv2d(conv, 256, (3,3), (1,1), scope='conv3_1', 
                  activation='none', wd=0.0005)
    conv = batch_norm(conv, 256, phase_train, scope='bn3_1')
    conv = tf.nn.elu(conv)
    conv = tf.nn.dropout(conv, tf.minimum(keep_prob+0.4, 1.))
    conv = conv2d(conv, 256, (3,3), (1,1), scope='conv3_2', 
                  activation='none', wd=0.0005)
    conv = batch_norm(conv, 256, phase_train, scope='bn3_2')
    conv = tf.nn.elu(conv)
    conv = tf.nn.dropout(conv, tf.minimum(keep_prob+0.4, 1.))
    conv = conv2d(conv, 256, (3,3), (1,1), scope='conv3_3', 
                  activation='none', wd=0.0005)
    conv = batch_norm(conv, 256, phase_train, scope='bn3_3')
    conv = tf.nn.elu(conv)
    conv = maxpool(conv, (2, 2), (2, 2))
    
    conv = conv2d(conv, 512, (3,3), (1,1), scope='conv4_1', 
                  activation='none', wd=0.0005)
    conv = batch_norm(conv, 512, phase_train, scope='bn4_1')
    conv = tf.nn.elu(conv)
    conv = tf.nn.dropout(conv, tf.minimum(keep_prob+0.4, 1.))
    conv = conv2d(conv, 512, (3,3), (1,1), scope='conv4_2', 
                  activation='none', wd=0.0005)
    conv = batch_norm(conv, 512, phase_train, scope='bn4_2')
    conv = tf.nn.elu(conv)
    conv = tf.nn.dropout(conv, tf.minimum(keep_prob+0.4, 1.))
    conv = conv2d(conv, 512, (3,3), (1,1), scope='conv4_3', 
                  activation='none', wd=0.0005)
    conv = batch_norm(conv, 512, phase_train, scope='bn4_4')
    conv = tf.nn.elu(conv)
    conv = maxpool(conv, (2, 2), (2, 2))
    
    conv = flatten(conv)
    conv = tf.nn.dropout(conv, tf.minimum(keep_prob+0.5, 1.))
    
    dense = fully_conn(conv, 512, scope='dense1', wd=0.0005)
    dense = tf.nn.elu(dense)
    
    dense = tf.nn.dropout(dense, tf.minimum(keep_prob+0.5, 1.))
    dense = fully_conn(dense, 512, scope='dense2', wd=0.0005)
    out = output(dense, 10, scope='dense3')
    #print(out.get_shape().as_list())
    return out

def block_op(x_tensor, conv_num_outputs, conv_ksize, conv_strides, phase_train,
             scope=None, padding='SAME', initialize='he_normal', wd=0.0):
    conv = conv2d(x_tensor, conv_num_outputs, conv_ksize, conv_strides,
                  padding=padding, scope=scope, activation='', wd=wd)
    conv = batch_norm(conv, conv_num_outputs, phase_train, scope=scope+'/bn')
    conv = tf.nn.elu(conv)
    return conv

def conv_net_v6(x, keep_prob):
    
    #Keep dropuout to 0 during training
    """
    Currently gives a performance of around 85% for 300 epochs
    
    """
        
    phase_train = tf.less(keep_prob, 1.)
    
    conv = block_op(x, 192, (5,5), (1,1), phase_train, scope='conv1_1',
                    padding='SAME', wd=0.0)
    conv = block_op(conv, 160, (1,1), (1,1), phase_train, scope='conv1_2',
                    padding='VALID',wd=0.0)
    conv = block_op(conv, 96, (1,1), (1,1), phase_train, scope='conv1_3',
                    padding='VALID',wd=0.0)
    
    conv = maxpool(conv, (3, 3), (2, 2))
    conv = tf.nn.dropout(conv, tf.minimum(keep_prob, 1.))
    
    conv = block_op(conv, 192, (5,5), (1,1), phase_train, scope='conv2_1',
                    padding='SAME',wd=0.0005)
    conv = block_op(conv, 192, (1,1), (1,1), phase_train, scope='conv2_2',
                    padding='VALID',wd=0.0005)
    conv = block_op(conv, 192, (1,1), (1,1), phase_train, scope='conv2_3',
                    padding='VALID',wd=0.0005)
    
    conv = avgpool(conv, (3, 3), (2, 2))
    conv = tf.nn.dropout(conv, tf.minimum(keep_prob, 1.))  
    
    conv = block_op(conv, 192, (3,3), (1,1), phase_train, scope='conv3_1',
                    padding='SAME',wd=0.0005)
    conv = block_op(conv, 192, (1,1), (1,1), phase_train, scope='conv3_2',
                    padding='VALID',wd=0.0005)
    conv = block_op(conv, 10, (1,1), (1,1), phase_train, scope='conv3_3',
                    padding='VALID',wd=0.0005)
    
    print(conv.get_shape().as_list())
    conv = avgpool(conv, (8, 8), (8, 8))
    print(conv.get_shape().as_list())
    out = flatten(conv)
    print(out.get_shape().as_list())
    return out

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer

# add all losses

with tf.name_scope('cross_entropy'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
tf.summary.scalar('cross_entropy', cost)
tf.add_to_collection('losses', cost)

cost = tf.add_n(tf.get_collection('losses'), name='total_loss')

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(3e-4).minimize(cost)
    #optimizer = tf.train.AdagradOptimizer(0.05).minimize(cost)
    #optimizer = tf.train.MomentumOptimizer(0.05, 0.9, use_nesterov=True).minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()

def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    # TODO: Implement Function
    #print('feature and label length', len(feature_batch),len(label_batch))
    session.run(optimizer, feed_dict={x:feature_batch, y:label_batch, keep_prob:keep_probability})

def print_stats(session, feature_batch, label_batch, cost, accuracy, epoch):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    # TODO: Implement Function
    
    summary, loss, acc = session.run([merged, cost, accuracy], feed_dict={x:feature_batch, y:label_batch, keep_prob:1.})
    train_writer.add_summary(summary, epoch)

    print('Loss: {:>10.4f} Accuracy: {:.6f}'.format(loss, acc))

epochs = 10
batch_size = 64
keep_probability = 0.5

save_model_path = './image_classification'

print('Training...')
import time
with tf.Session() as sess:
    
    
    train_writer = tf.summary.FileWriter('train', sess.graph)
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    t1 = time.time()
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            
            if epoch % 10 ==0:
                t2 = time.time()
                print('Epoch {:>2}, CIFAR-10 Batch {}: Time: {:.2f} '.format(epoch + 1, batch_i, t2-t1), end='')
                print_stats(sess, batch_features, batch_labels, cost, accuracy, epoch)
                t1 = time.time()
            
    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)
# Set batch size if not already set
try:
    if batch_size:
        pass
except NameError:
    batch_size = 64

save_model_path = './image_classification'
n_samples = 4
top_n_predictions = 3

import random

tf.reset_default_graph()
def test_model():
    """
    Test the saved model against the test dataset
    """

    test_features, test_labels = pickle.load(open('preprocess_training.p', mode='rb'))
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
        
        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0
        
        for train_feature_batch, train_label_batch in helper.batch_features_labels(test_features, test_labels, batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: train_feature_batch, loaded_y: train_label_batch, loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))

        # Print Random Samples
        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))
        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels, loaded_keep_prob: 1.0})
        helper.display_image_predictions(random_test_features, random_test_labels, random_test_predictions)


test_model()
