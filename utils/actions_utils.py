import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import imresize
from operator import itemgetter
import cv2
import pdb

# actions imshow convenience function
def actions_imshow(img,im_size):
    plt.imshow(img.reshape([im_size,im_size,3]))
    plt.axis('off')

# load Stanford-40 Actions dataset
def load_actions(path, inp_size):
    # read filenames and labels
    fid = open(path+"images.txt","r")
    img_names = fid.read().splitlines()
    fid.close()
    fid = open(path+"labels.txt","r")
    lbl_names = fid.read().splitlines()
    fid.close()
    fid = open(path+"splits.txt","r")
    spl_names = fid.read().splitlines()
    fid.close()

    # parse splits
    splits = []
    for m in xrange(len(spl_names)):
        splits.append(int(spl_names[m]))
    # parse labels
    trn_lbl = []
    val_lbl = []
    tst_lbl = []
    for m in xrange(len(lbl_names)):
        if splits[m]==3:
            tst_lbl.append(int(lbl_names[m])-1)
        else:
            if splits[m]==2:
                val_lbl.append(int(lbl_names[m])-1)
            else:
                trn_lbl.append(int(lbl_names[m])-1)
    # parse images
    trn_img = []
    val_img = []
    tst_img = []
    for m in xrange(len(img_names)):
        # read the image
        data = cv2.imread(path+"JPEGImages/"+img_names[m])
        #data = np.asarray(data)
        if len(data.shape)==2:
            data = np.repeat(data[:,:, np.newaxis], 3, axis=2)
        data = imresize(data,(inp_size, inp_size, 3))
        #pdb.set_trace()
        # add it to the corresponding split
        if splits[m]==3:
            tst_img.append(data)
        else:
            if splits[m]==2:
                val_img.append(data)
            else:
                trn_img.append(data)
    
    return trn_img, val_img, tst_img, trn_lbl, val_lbl, tst_lbl

# return a new actions dataset
def disjoint_actions(actions,nums):
    pos_trn = []
    for i in range(len(nums)):
        tmp = np.where(np.asarray(actions[3]) == nums[i])[0]
        pos_trn = np.hstack((pos_trn,tmp))
        pos_trn  = np.asarray(pos_trn).astype(int)
        np.random.shuffle(pos_trn)
    pos_tst = []
    for i in range(len(nums)):
        tmp = np.where(np.asarray(actions[5]) == nums[i])[0]
        pos_tst = np.hstack((pos_tst,tmp))
        pos_tst  = np.asarray(pos_tst).astype(int)
        np.random.shuffle(pos_tst)
    
    trn_img = itemgetter(*pos_trn)(actions[0])
    val_img = actions[1]
    tst_img = itemgetter(*pos_tst)(actions[2])
    trn_lbl = itemgetter(*pos_trn)(actions[3])
    val_lbl = actions[4]
    tst_lbl = itemgetter(*pos_tst)(actions[5])
    
    return trn_img, val_img, tst_img, trn_lbl, val_lbl, tst_lbl

# get equally distributed samples among given classes from a split
def get_ed_samples(data, samples=10):
    # retrieve number of samples for each class
    indx = []
    classes = np.unique(data.labels)
    for cl in range(len(classes)):
        tmp = np.where(data.labels == classes[cl])[0]
        np.random.shuffle(tmp)
        indx = np.hstack((indx,tmp[0:np.min(samples, len(tmp))]))
        indx = np.asarray(indx).astype(int)
        
    return indx
