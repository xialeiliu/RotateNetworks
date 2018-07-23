import os
import cPickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from utils import birds_utils
from utils import utils_vgg16
from params import Params

######### Modifiable Settings ##########
gpu         = Params.gpu_ewc      # Used GPU
nb_cl       = Params.nb_cl        # Classes per group
nb_gr       = Params.nb_groups    # Number of groups
nb_sa       = Params.num_samples  # for calculating Fisher Information
lr_init     = Params.lr_init      # Starting learning rate
wght_decay  = Params.wght_decay   # Weight Decay
train_path  = Params.train_path   # Path to read images
data_size   = Params.data_size    # Image size
ratio       = Params.ratio        # Trade-off between old tasks and new task
eval_single = Params.eval_single  # True: Evaluate different task separately
########################################

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

# Sanity Initializations
model_vgg = []
acc_task = np.zeros((nb_gr, nb_gr))

# Loading CUB-200 Birds dataset
print 'Loading full dataset...'
full_birds = birds_utils.load_birds(train_path, data_size)

# Experiment name: results will be loaded from this folder.
Ex_name = 'EWC_vgg16' + str(nb_gr) + '_lr' + str(lr_init) + '_sample_' + str(nb_sa) + '_ratio_' + str(ratio) + '/'
save_path = Params.save_path + Ex_name
print 'Files will be loaded from: ' + save_path
print '---'

# Evaluate performance
for it in range(nb_gr):

    # Initialization
    confusion_mat = []
    for _ in range(2):
        confusion_mat.append([])

    # Define placeholders
    x = tf.placeholder("float", [None, data_size, data_size, 3])
    y = tf.placeholder("float", [None, nb_cl * nb_gr])

    # Get the data for this task
    print('Retrieving data for task {}...'.format(it + 1))
    _, _, tst_img, _, _, tst_lbl = birds_utils.disjoint_birds(full_birds, range(0, (it+1) * nb_cl))

    # Building the graph
    if it == 0:
        with tf.variable_scope('Vgg16'):
            model_vgg = utils_vgg16.Vgg16(x)
    else:
        with tf.variable_scope('Vgg16'):
            model_vgg.build(x, rotate=False, pre_init=True)
    correct_pred = tf.equal(tf.argmax(model_vgg.y, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Loading weights
    save_weights = cPickle.load(open(save_path + 'model-iteration' + str(nb_cl) + '-%i.pickle' % it, 'rb'))

    # Inference and metrics
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        variables_graph = tf.get_collection(tf.GraphKeys.WEIGHTS)
        void0 = sess.run([(variables_graph[i]).assign(save_weights[i]) for i in range(len(variables_graph))])

        fg = np.arange(len(tst_img))
        # Evaluation routine
        for i in range(int(np.floor(len(tst_img)))):
            batch_x = tst_img[fg[i:(i+1)]]
            batch_y = tst_lbl[fg[i:(i+1)]]
            acc, sc = sess.run([accuracy, model_vgg.y], feed_dict={x: batch_x, y: np.eye(nb_gr*nb_cl)[batch_y]})
            tmp = np.argsort(sc[:, batch_y[0]/nb_cl*nb_cl:(batch_y[0]/nb_cl+1)*nb_cl], axis=1)[:, -1:]   # Top 1 accuracy
            if eval_single:
                # Single-head evaluation
                confusion_mat[0].append(batch_y)
                confusion_mat[1].append(np.argsort(sc[:, :(it+1)*nb_cl], axis=1)[:, -1:]) 
            else:
                # Multi-head evaluation
                confusion_mat[0].append(batch_y)
                confusion_mat[1].append(tmp+batch_y/nb_cl*nb_cl)

        confusion_mat = np.asarray(confusion_mat)
        confusion_mat = np.reshape(confusion_mat, (confusion_mat.shape[0], confusion_mat.shape[1]))
        for k in range(it+1):
            cmat = confusion_matrix(confusion_mat[0], confusion_mat[1])
            tmp = cmat.diagonal()/(cmat.sum(axis=1)*1.0)
            acc_task[it, k] = np.sum(tmp[k*nb_cl:(k+1)*nb_cl]/nb_cl)
        
    tf.reset_default_graph()

print('Accuracy for each task is:' + str(acc_task))
