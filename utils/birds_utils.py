import numpy as np
from scipy.misc import imresize
from operator import itemgetter
import cv2


# load CUB200-2011 Birds dataset
def load_birds(path, inp_size):
    # read filenames and labels
    fid = open(path+"images.txt", "r")
    img_names = fid.read().splitlines()
    fid.close()
    fid = open(path+"image_class_labels.txt", "r")
    lbl_names = fid.read().splitlines()
    fid.close()
    fid = open(path+"train_test_split.txt", "r")
    spl_names = fid.read().splitlines()
    fid.close()
    fid = open(path+"bounding_boxes.txt", "r")
    bounding_boxes = fid.read().splitlines()
    fid.close()
    # parse splits
    splits = []
    for m in xrange(len(spl_names)):
        splits.append(int(spl_names[m].split()[1]))
    # parse labels
    trn_lbl = []
    val_lbl = []
    tst_lbl = []
    for m in xrange(len(lbl_names)):
        if splits[m] == 0:
            tst_lbl.append(int(lbl_names[m].split()[1])-1)
        else:
            trn_lbl.append(int(lbl_names[m].split()[1])-1)
    # parse images
    trn_img = []
    val_img = []
    tst_img = []
    for m in xrange(len(img_names)):
        # read the image
        data = cv2.imread(path+"images/"+img_names[m].split()[1])
        # data = np.asarray(data)
        if len(data.shape) == 2:
            data = np.repeat(data[:, :, np.newaxis], 3, axis=2)
        bbox = np.int32(np.float32(bounding_boxes[m].split())[1:5])
        data = data[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2], :]
        data = imresize(data, (inp_size, inp_size, 3))
        # add it to the corresponding split
        if splits[m] == 0:
            tst_img.append(data)
        else:
            trn_img.append(data)
    
    return trn_img, val_img, tst_img, trn_lbl, val_lbl, tst_lbl


# return a new birds dataset
def disjoint_birds(birds, nums):
    pos_trn = []
    for i in range(len(nums)):
        tmp = np.where(np.asarray(birds[3]) == nums[i])[0]
        pos_trn = np.hstack((pos_trn, tmp))
        pos_trn = np.asarray(pos_trn).astype(int)
        np.random.shuffle(pos_trn)
    pos_tst = []
    for i in range(len(nums)):
        tmp = np.where(np.asarray(birds[5]) == nums[i])[0]
        pos_tst = np.hstack((pos_tst, tmp))
        pos_tst = np.asarray(pos_tst).astype(int)
        np.random.shuffle(pos_tst)
    
    trn_img = np.asarray(itemgetter(*pos_trn)(birds[0]))
    val_img = birds[1]
    tst_img = np.asarray(itemgetter(*pos_tst)(birds[2]))
    trn_lbl = np.asarray(itemgetter(*pos_trn)(birds[3])).astype(int)
    val_lbl = birds[4]
    tst_lbl = np.asarray(itemgetter(*pos_tst)(birds[5])).astype(int)
    
    return trn_img, val_img, tst_img, trn_lbl, val_lbl, tst_lbl
