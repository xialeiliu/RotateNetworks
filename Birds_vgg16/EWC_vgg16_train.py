import os
import time
import cPickle
import numpy as np
import tensorflow as tf

from utils import utils_vgg16
from utils import birds_utils
from params import Params

######### Modifiable Settings ##########
gpu        = Params.gpu_ewc      # Used GPU
nb_cl      = Params.nb_cl        # Classes per group
nb_gr      = Params.nb_groups    # Number of groups
nb_sa      = Params.num_samples  # for calculating Fisher Information
batch_size = Params.batch_size   # Batch size
epochs     = Params.epochs       # Total number of epochs
lr_init    = Params.lr_init      # Starting learning rate
lr_strat   = Params.lr_strat     # Epochs where learning rate gets decreased
lr_factor  = Params.lr_factor    # Learning rate decrease factor
wght_decay = Params.wght_decay   # Weight Decay
train_path = Params.train_path   # Path to read images
data_size  = Params.data_size    # Image size
ratio      = Params.ratio        # Trade-off between old tasks and new task
########################################

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

# Sanity Initializations
save_weights, variables_graph, model_vgg, pre_img, pre_lbl = [], [], [], [], []

# Loading CUB-200 Birds dataset
print 'Loading full dataset...'
full_birds = birds_utils.load_birds(train_path, data_size)

# Experiment name: results will be saved in this folder.
Ex_name = 'EWC_vgg16' + str(nb_gr) + '_lr' + str(lr_init) + '_sample_' + str(nb_sa) + '_ratio_' + str(ratio) + '/'
save_path = Params.save_path + Ex_name
try:
    os.stat(save_path)
except:
    os.mkdir(save_path)
print 'Files will be saved to: ' + save_path
print '---'

# Learn one task at a time
for it in range(nb_gr):

    # Define placeholders
    x = tf.placeholder("float", [None, data_size, data_size, 3])
    y = tf.placeholder("float", [None, nb_cl*nb_gr])
    learning_rate = tf.placeholder(tf.float32, shape=[])

    if it == 0:
        # Build the graph with loss function
        print('Building initial model...')
        with tf.variable_scope('Vgg16'):
            model_vgg = utils_vgg16.Vgg16(x)
            variables_graph = tf.get_collection(tf.GraphKeys.WEIGHTS)
            new_cl = range(it*nb_cl, nb_gr*nb_cl)
            label_new_classes = tf.stack([y[:, i] for i in new_cl], axis=1)
            pred_new_classes = tf.stack([model_vgg.y[:, i] for i in new_cl], axis=1)
            model_vgg.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_new_classes,
                                                                                             logits=pred_new_classes))
        model_vgg.set_vanilla_loss(learning_rate, wght_decay, variables_graph)

    if it > 0:
        # Compute the Fisher Information of the previous task before learning a new task
        print('Computing Fisher Information of task {}...'.format(it))
        with tf.Session(config=config) as sess:
            with tf.variable_scope('Vgg16'):
                model_vgg.build(x, rotate=False, pre_init=True)
                sess.run(tf.global_variables_initializer())
                variables_graph = tf.get_collection(tf.GraphKeys.WEIGHTS)
                void0 = sess.run([(variables_graph[i]).assign(save_weights[i]) for i in range(len(variables_graph))])
                model_vgg.var_list = [v for v in tf.trainable_variables()]
                model_vgg.star()
                model_vgg.compute_fisher(pre_img, pre_lbl, sess, num_samples=nb_sa, eq_distrib=True)
        tf.reset_default_graph()

        # Update the graph with loss function
        print('Updating loss function of task {}...'.format(it))
        x = tf.placeholder("float", [None, data_size, data_size, 3])
        y = tf.placeholder("float", [None, nb_cl*nb_gr])
        learning_rate = tf.placeholder(tf.float32, shape=[])
        with tf.variable_scope('Vgg16'):
            model_vgg.build(x, rotate=False, pre_init=True)
            new_cl = range(it*nb_cl, nb_gr*nb_cl)
            label_new_classes = tf.stack([y[:, i] for i in new_cl], axis=1)
            pred_new_classes = tf.stack([model_vgg.y[:, i] for i in new_cl], axis=1)
            model_vgg.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_new_classes,
                                                                                             logits=pred_new_classes))
            variables_graph = tf.get_collection(tf.GraphKeys.WEIGHTS)
            trainable_list = [v for v in tf.trainable_variables()]
            mean_list = [v for v in variables_graph if "mean" in v.name]
            var_list = [v for v in variables_graph if "var" in v.name]
            model_vgg.var_list = trainable_list
        model_vgg.update_ewc_loss(learning_rate, ratio, wght_decay, trainable_list+mean_list+var_list, it*nb_cl)

    # Get the data for this task
    print('Retrieving data for task {}...'.format(it + 1))
    trn_img, _, _, trn_lbl, _, _ = birds_utils.disjoint_birds(full_birds, range(it * nb_cl, (it + 1) * nb_cl))

    # Run the learning phase
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        lr = lr_init
        # Except for first task, initialize with weights learned from previous tasks
        if it > 0:
            void0 = sess.run([(variables_graph[i]).assign(save_weights[i]) for i in range(len(variables_graph))])
        # Training
        for epoch in range(epochs):
            loss, accuracy = [], []
            epoch_time = time.time()
            fg = np.random.permutation(len(trn_img))
            for i in range(int(np.floor(len(trn_img)/batch_size))):
                batch_x = trn_img[fg[i*batch_size:(i+1)*batch_size]]
                batch_y = trn_lbl[fg[i*batch_size:(i+1)*batch_size]]
                inp_dict = {x: batch_x, y: np.eye(nb_gr*nb_cl)[batch_y], learning_rate: lr}
                loss_batch, _, sc = sess.run([model_vgg.loss, model_vgg.train_step, model_vgg.y], feed_dict=inp_dict)
                loss.append(loss_batch)
                accuracy = np.mean(np.equal(np.argmax(sc, 1), batch_y))
            # Decrease the learning rate when scheduled
            if epoch in lr_strat:
                lr /= lr_factor
            print("Task {} Epoch {}: accuracy {} -- loss {} -- time {}".format(it, epoch, accuracy, np.mean(loss),
                                                                               time.time() - epoch_time))

        # copy weights to store network
        model_vgg.var_list = [v for v in variables_graph]
        save_weights = sess.run([model_vgg.var_list[i] for i in range(len(model_vgg.var_list))])
        cPickle.dump(save_weights, open(save_path + 'model-iter' + str(nb_cl) + '-%i.pickle' % it, 'w'))
    pre_img, pre_lbl = trn_img, trn_lbl  # used to calculate the Fisher Information Matrix
    tf.reset_default_graph()
