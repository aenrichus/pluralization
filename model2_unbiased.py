import tensorflow as tf
import pandas as pd
import numpy as np
import random


theory = 'min_x'


# function to initialize weights with noise
def weight_variable(shape):
    initial = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# function to initialize with zero biases
def bias_variable(shape):
    initial = tf.zeros(shape)
    return tf.Variable(initial)


print("Reading files...")
plurals = pd.read_table("plurals1.txt", encoding='utf-16')

print("Preparing arrays...")
feat_input = np.zeros((len(plurals), 31))  # features for 60 words (30 item features + 1 -pl feature)
vins_layer = np.zeros((len(plurals), 8*27+1))  # vocab insertion (30 sing. (+ 10 irr. pl.) + 1 -pl marker)
phon_output = np.zeros((len(plurals), 8*27+1))  # phonology (8 slots + 1 final -s)

print("Creating lists...")
chars = list('_abcedfghijklmnopqrstuvwxyz')
stems = list(plurals['STEM'].unique())  # generates a list of all unique stems

print("Padding representations...")
# pad short word stems with underscores
padded_stems = []
for i in plurals['STEM']:
    while len(list(i)) < 8:
        i = '_' + i
    padded_stems.append(i)

# pad short word phons with underscores; create min_x values
padded_phons = []
padded_min_x = []
first_forty = 0
for i in plurals['WORD']:
    if i[-1] == 's':
        i = i[:-1]
    while len(list(i)) < 8:
        i = '_' + i
    padded_phons.append(i)
    if first_forty < 40:
        padded_min_x.append(i)
        first_forty += 1
padded_min_x += padded_stems[40:]

print("Generating layers...")
for i in range(0, len(plurals)):
    feat_input[i] = np.concatenate([np.eye(30)[stems.index(plurals['STEM'][i])],
                                    np.array([plurals['VERSION'][i]])])

    # implement the two theories based on the vocab insertion representation
    if theory == 'rules':
        vins_layer[i] = np.concatenate([np.eye(27)[chars.index(padded_stems[i][0])],
                                        np.eye(27)[chars.index(padded_stems[i][1])],
                                        np.eye(27)[chars.index(padded_stems[i][2])],
                                        np.eye(27)[chars.index(padded_stems[i][3])],
                                        np.eye(27)[chars.index(padded_stems[i][4])],
                                        np.eye(27)[chars.index(padded_stems[i][5])],
                                        np.eye(27)[chars.index(padded_stems[i][6])],
                                        np.eye(27)[chars.index(padded_stems[i][7])],
                                        np.array([plurals['VERSION'][i]])])
    elif theory == 'min_x':
        vins_layer[i] = np.concatenate([np.eye(27)[chars.index(padded_min_x[i][0])],
                                        np.eye(27)[chars.index(padded_min_x[i][1])],
                                        np.eye(27)[chars.index(padded_min_x[i][2])],
                                        np.eye(27)[chars.index(padded_min_x[i][3])],
                                        np.eye(27)[chars.index(padded_min_x[i][4])],
                                        np.eye(27)[chars.index(padded_min_x[i][5])],
                                        np.eye(27)[chars.index(padded_min_x[i][6])],
                                        np.eye(27)[chars.index(padded_min_x[i][7])],
                                        np.array([plurals['VERSION'][i]])])

    phon_output[i] = np.concatenate([np.eye(27)[chars.index(padded_phons[i][0])],
                                     np.eye(27)[chars.index(padded_phons[i][1])],
                                     np.eye(27)[chars.index(padded_phons[i][2])],
                                     np.eye(27)[chars.index(padded_phons[i][3])],
                                     np.eye(27)[chars.index(padded_phons[i][4])],
                                     np.eye(27)[chars.index(padded_phons[i][5])],
                                     np.eye(27)[chars.index(padded_phons[i][6])],
                                     np.eye(27)[chars.index(padded_phons[i][7])],
                                     np.array([plurals['ADD_S'][i]])])

representations = list(zip(feat_input, vins_layer, phon_output))

# implementing the graph
x = tf.placeholder("float", shape=[None, len(feat_input[0])])
v_ = tf.placeholder("float", shape=[None, len(vins_layer[0])])
y_ = tf.placeholder("float", shape=[None, len(phon_output[0])])

# implement feat -> vocab ins
W_1 = weight_variable([len(feat_input[0]), len(vins_layer[0])])
l_1 = tf.matmul(x, W_1)  # multiply by weights and add bias
v = tf.nn.sigmoid(l_1)  # apply sigmoid

# calculate error
first_cost = tf.nn.sigmoid_cross_entropy_with_logits(l_1, v_)  # cost function: sigmoid cross entropy with logits

# implement the output layer
W_2 = weight_variable([len(vins_layer[0]), len(phon_output[0])])
l_2 = tf.matmul(v, W_2)
y = tf.nn.sigmoid(l_2)

# training
cost = tf.nn.sigmoid_cross_entropy_with_logits(l_2, y_)  # cost function: sigmoid cross entropy with logits
train_step = tf.train.GradientDescentOptimizer(0.1).minimize((cost + first_cost))  # sets learning rates

# evaluation
sse = tf.reduce_sum(tf.square(y - y_))  # calculate SSE (not used in training)

predicted = tf.round(y)  # converts to 0s and 1s by splitting at 0.5
correct_activations = tf.equal(predicted, y_)  # check that the predicted and real values are equal
each_correct = tf.reduce_all(correct_activations, 1)  # itemwise comparison that all are equal
accuracy = tf.reduce_mean(tf.cast(each_correct, tf.float32))  # convert to float and take the mean

print("Setting variables...")
n_epochs = 100000  # total number of words to present
check_freq = 100  # how often to check accuracy
train_accuracies = np.zeros(int(n_epochs/check_freq))  # placeholder for graphing accuracy
test_accuracies = np.zeros(int(n_epochs/check_freq))

print("Beginning training...")
for j in range(1):  # runs through the model using each value of hidden units
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        test_x, test_v, test_y = zip(*representations)

        for i in range(n_epochs+1):
            batch_x, batch_v, batch_y = zip(*random.sample(representations, 1))

            if i % check_freq == 0:  # in order to print the result
                err, acc = sess.run([sse, accuracy], feed_dict={x: test_x, v_: test_v, y_: test_y})
                print("step %s - sse: %s - acc: %s" % (i, err, acc))

            train_step.run(feed_dict={x: batch_x, v_: batch_v, y_: batch_y})

print("Game over?")
