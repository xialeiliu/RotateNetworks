import numpy as np
import cPickle
import os
import pdb
import cv2

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_data(train_path,order,nb_groups, nb_cl, nb_val,SubMean = False):
    xs = []
    ys = []
    for j in range(1):
      d = unpickle(train_path+'cifar-100-python/train')
      x = d['data']
      y = d['fine_labels']
      xs.append(x)
      ys.append(y)
    
    d = unpickle(train_path + 'cifar-100-python/test')
    xs.append(d['data'])
    ys.append(d['fine_labels'])
    
    x = np.concatenate(xs)/np.float32(255)
    y = np.concatenate(ys)
    #x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 3,32, 32)).transpose(0,2,3,1)
    #x = np.transpose(x,(0,2,3,1))
    #pdb.set_trace()
    #cv2.imwrite("1.jpg",cv2.cvtColor(x[3,:,:,:]*255, cv2.COLOR_RGB2BGR))
    #.transpose(0,3,1,2)
    # subtract per-pixel mean
    pixel_mean = np.mean(x[0:50000],axis=0)
    #np.save('cifar_mean.npy',pixel_mean)
    #pdb.set_trace()
    if SubMean == True:
        x -= pixel_mean
    #pdb.set_trace()
    # Create Train/Validation set
    eff_samples_cl = 500-nb_val
    X_train = np.zeros((eff_samples_cl*100,32, 32,3))
    Y_train = np.zeros(eff_samples_cl*100)
    X_valid = np.zeros((nb_val*100,32, 32,3))
    Y_valid = np.zeros(nb_val*100)
    for i in range(100):
        index_y=np.where(y[0:50000]==i)[0]
        np.random.shuffle(index_y)
        X_train[i*eff_samples_cl:(i+1)*eff_samples_cl] = x[index_y[0:eff_samples_cl],:,:,:]
        Y_train[i*eff_samples_cl:(i+1)*eff_samples_cl] = y[index_y[0:eff_samples_cl]]
        X_valid[i*nb_val:(i+1)*nb_val] = x[index_y[eff_samples_cl:500],:,:,:]
        Y_valid[i*nb_val:(i+1)*nb_val] = y[index_y[eff_samples_cl:500]]
    
    X_test  = x[50000:,:,:,:]
    Y_test  = y[50000:]
   
    files_train = []
    train_labels = []
    files_valid = []
    valid_labels = []
    files_test = []
    test_labels = []
    
    
    for _ in range(nb_groups):
      files_train.append([])
      train_labels.append([])
      files_valid.append([])
      valid_labels.append([])
      files_test.append([])
      test_labels.append([])

    for i in range(nb_groups):
      for i2 in range(nb_cl):
        labels_old = Y_train
        #pdb.set_trace()
        tmp_ind=np.where(labels_old == order[nb_cl*i+i2])[0]
        np.random.shuffle(tmp_ind)
        files_train[i].extend(X_train[tmp_ind[0:len(tmp_ind)]])
        train_labels[i].extend(Y_train[tmp_ind[0:len(tmp_ind)]])

        labels_old = Y_valid
        tmp_ind=np.where(labels_old == order[nb_cl*i+i2])[0]
        np.random.shuffle(tmp_ind)
        files_valid[i].extend(X_valid[tmp_ind[0:len(tmp_ind)]])
        valid_labels[i].extend(Y_valid[tmp_ind[0:len(tmp_ind)]])
       

        labels_old = Y_test
        tmp_ind=np.where(labels_old == order[nb_cl*i+i2])[0]
        np.random.shuffle(tmp_ind)
        files_test[i].extend(X_test[tmp_ind[0:len(tmp_ind)]])
        test_labels[i].extend(Y_test[tmp_ind[0:len(tmp_ind)]])
    #pdb.set_trace()
    return files_train,train_labels,files_valid,valid_labels,files_test,test_labels

def aug(batch):
    # as in paper : 
    # pad feature arrays with 4 pixels on each side
    # and do random cropping of 32x32
    #pdb.set_trace()
    padded = np.pad(batch,((0,0),(4,4),(4,4),(0,0)),mode='constant')
    random_cropped = np.zeros(batch.shape, dtype=np.float32)
    crops = np.random.random_integers(0,high=8,size=(batch.shape[0],2))
    for r in range(batch.shape[0]):
        # Cropping and possible flipping
        #if (np.random.randint(2) > 0):
        random_cropped[r,:,:,:] = padded[r,crops[r,0]:(crops[r,0]+32),crops[r,1]:(crops[r,1]+32),:]
        #else:
            
        #random_cropped[r,:,:,:] = padded[r,crops[r,0]:(crops[r,0]+32),crops[r,1]:(crops[r,1]+32),:][:,:,::-1]
    inp_exc = random_cropped

    return inp_exc
        
def balanced_subsample(x,y,subsample_size=1.0):

    class_xs = []
    min_elems = None
    #pdb.set_trace()
    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []
    #pdb.set_trace()
    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys
