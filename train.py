
import csv
import re
import random
import numpy as np

from IPython import embed

#Separates a file with mixed positive and negative examples into two.
def separate_dataset(filename):
    good_out = open("good_"+filename,"w+");
    bad_out  = open("bad_"+filename,"w+");

    seen = 1;
    with open(filename,'r') as f:
        reader = csv.reader(f)
        reader.next()

        for line in reader:
            seen +=1
            sentiment = line[1]
            sentence = line[3]

            if (sentiment == "0"):
                bad_out.write(sentence+"\n")
            else:
                good_out.write(sentence+"\n")

            if (seen%10000==0):
                print (seen);

    good_out.close();
    bad_out.close();



#Load Dataset
def get_dataset(goodfile,badfile,limit,randomize=True):
    good_x = list(open(goodfile,"r").readlines())
    good_x = [s.strip() for s in good_x]
    
    bad_x  = list(open(badfile,"r").readlines())
    bad_x  = [s.strip() for s in bad_x]

    if (randomize):
        random.shuffle(bad_x)
        random.shuffle(good_x)

    good_x = good_x[:limit]
    bad_x = bad_x[:limit]

    x = good_x + bad_x
    x = [clean_str(s) for s in x]


    positive_labels = [[0, 1] for _ in good_x]
    negative_labels = [[1, 0] for _ in bad_x]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x,y]




#Clean Dataset
def clean_str(string):


    #EMOJIS
    string = re.sub(r":\)","emojihappy1",string)
    string = re.sub(r":P","emojihappy2",string)
    string = re.sub(r":p","emojihappy3",string)
    string = re.sub(r":>","emojihappy4",string)
    string = re.sub(r":3","emojihappy5",string)
    string = re.sub(r":D","emojihappy6",string)
    string = re.sub(r" XD ","emojihappy7",string)
    string = re.sub(r" <3 ","emojihappy8",string)

    string = re.sub(r":\(","emojisad9",string)
    string = re.sub(r":<","emojisad10",string)
    string = re.sub(r":<","emojisad11",string)
    string = re.sub(r">:\(","emojisad12",string)

    #MENTIONS "(@)\w+"
    string = re.sub(r"(@)\w+","mentiontoken",string)
    
    #WEBSITES
    string = re.sub(r"http(s)*:(\S)*","linktoken",string)

    #STRANGE UNICODE \x...
    string = re.sub(r"\\x(\S)*","",string)

    #General Cleanup and Symbols
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()



#Generate random batches
#Source: https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
def gen_batch(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

#if __name__ == "__main__":
#    separate_dataset("small.txt");


#42
#642
            
            
import tensorflow as tf
import numpy as np
from IPython import embed

class LSTM_CNN(object):
    def LSTM_CNN(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0,num_hidden=100):

        # PLACEHOLDERS
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")    # X - The Data
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")      # Y - The Lables
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")       # Dropout

        
        l2_loss = tf.constant(0.0) # Keeping track of l2 regularization loss

        #1. EMBEDDING LAYER ################################################################
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            #self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)


        #2. LSTM LAYER ######################################################################
        self.lstm_cell = tf.contrib.rnn.LSTMCell(32,state_is_tuple=True)
        #self.h_drop_exp = tf.expand_dims(self.h_drop,-1)
        self.lstm_out,self.lstm_state = tf.nn.dynamic_rnn(self.lstm_cell,self.embedded_chars,dtype=tf.float32)
        #embed()

        self.lstm_out_expanded = tf.expand_dims(self.lstm_out, -1)

        #2. CONVOLUTION LAYER + MAXPOOLING LAYER (per filter) ###############################
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # CONVOLUTION LAYER
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(self.lstm_out_expanded, W,strides=[1, 1, 1, 1],padding="VALID",name="conv")
                # NON-LINEARITY
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # MAXPOOLING
                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)

        # COMBINING POOLED FEATURES
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        
        # #3. DROPOUT LAYER ###################################################################
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


        print( "(!!) LOADED LSTM-CNN! :)")
        #embed()



# 1. Embed --> LSTM
# 2. LSTM --> CNN
# 3. CNN --> Pooling/Output



# 1. Embed --> LSTM
# 2. LSTM --> CNN
# 3. CNN --> Pooling/Output





#! /usr/bin/env python
import sys

#SELECT WHICH MODEL YOU WISH TO RUN:
#from lstm_cnn import LSTM_CNN  


import tensorflow as tf
import numpy as np
import os
import time
import datetime
#import batchgen
from tensorflow.contrib import learn

from IPython import embed

# Parameters
# ==================================================

# Data loading params
dev_size = .10

# Model Hyperparameters
embedding_dim  = 32     #128
max_seq_legth = 70 
filter_sizes = [3,4,5]  #3
num_filters = 32
dropout_prob = 0.5 #0.5
l2_reg_lambda = 0.0
use_glove = True #Do we use glove

# Training parameters
batch_size = 128
num_epochs = 10 #200
evaluate_every = 100 #100
checkpoint_every = 100000 #100
num_checkpoints = 0 #Checkpoints to store


# Misc Parameters
allow_soft_placement = True
log_device_placement = False



# Data Preparation
# ==================================================


filename = ""
goodfile = "good_tweets.csv"
badfile = "bad_tweets.csv"


# Load data
print("Loading data...")
x_text, y = get_dataset(goodfile, badfile, 5000) #TODO: MAX LENGTH

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
if (1):#not use_glove
    print ("Not using GloVe")
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
else:
    print ("Using GloVe")
    embedding_dim = 50
    filename = '../glove.6B.50d.txt'
    def loadGloVe(filename):
        vocab = []
        embd = []
        file = open(filename,'r')
        for line in file.readlines():
            row = line.strip().split(' ')
            vocab.append(row[0])
            embd.append(row[1:])
        print('Loaded GloVe!')
        file.close()
        return vocab,embd
    vocab,embd = loadGloVe(filename)
    vocab_size = len(vocab)
    embedding_dim = len(embd[0])
    embedding = np.asarray(embd)

    W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                    trainable=False, name="W")
    embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
    embedding_init = W.assign(embedding_placeholder)

    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})

    from tensorflow.contrib import learn
    #init vocab processor
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    #fit the vocab from glove
    pretrain = vocab_processor.fit(vocab)
    #transform inputs
    x = np.array(list(vocab_processor.transform(x_text)))

    #init vocab processor
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    #fit the vocab from glove
    pretrain = vocab_processor.fit(vocab)
    #transform inputs
    x = np.array(list(vocab_processor.transform(x_text)))


# Randomly shuffle data
np.random.seed(42)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(dev_size * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

#embed()


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        model = LSTM_CNN(x_train.shape[1],y_train.shape[1],len(vocab_processor.vocabulary_),embedding_dim,filter_sizes,num_filters,l2_reg_lambda)
     


        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", model.loss)
        acc_summary = tf.summary.scalar("accuracy", model.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        #TRAINING STEP
        def train_step(x_batch, y_batch,save=False):
            feed_dict = {
              model.input_x: x_batch,
              model.input_y: y_batch,
              model.dropout_keep_prob: dropout_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, model.loss, model.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if save:
                train_summary_writer.add_summary(summaries, step)

        #EVALUATE MODEL
        def dev_step(x_batch, y_batch, writer=None,save=False):
            feed_dict = {
              model.input_x: x_batch,
              model.input_y: y_batch,
              model.dropout_keep_prob: 0.5
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, model.loss, model.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if save:
                if writer:
                    writer.add_summary(summaries, step)

        #CREATE THE BATCHES GENERATOR
        batches = gen_batch(list(zip(x_train, y_train)), batch_size, num_epochs)
        
        #TRAIN FOR EACH BATCH
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
        dev_step(x_dev, y_dev, writer=dev_summary_writer)           
            
            
            
            
            
            
            
            
            