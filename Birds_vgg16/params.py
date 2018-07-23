class Params:
    # -------------------------------
    # GENERAL
    # -------------------------------
    gpu_ewc = '4'              # GPU number
    gpu_rewc = '3'             # GPU number
    data_size = 224            # Data size
    batch_size = 32            # Batch size
    nb_cl      = 50            # Classes per group 
    nb_groups  = 4             # Number of groups     200 = 50*4
    nb_val = 0                 # Number of images for val
    epochs     = 50            # Total number of epochs 
    num_samples = 5            # Samples per class to compute the Fisher information
    lr_init    =  0.001        # Initial learning rate
    lr_strat   = [40,80]       # Epochs where learning rate gets decreased
    lr_factor  = 5.            # Learning rate decrease factor
    wght_decay = 0.00001       # Weight Decay
    ratio = 100.0              # ratio = lwf loss / softmax loss

    # -------------------------------
    # Parameters for test
    # -------------------------------
    eval_single = True         # Evaluate different task in Single head setting

    # -------------------------------
    # Parameters for path
    # -------------------------------
    save_path   = './checkpoints/'  # Model saving path
    train_path  =  '/data/Datasets/CUB_200_2011/CUB_200_2011/' # Data path
