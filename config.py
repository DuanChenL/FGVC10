import os

_PATH = r'./'
PROJECT_ROOT = os.getcwd()
PATH = {
    'root': _PATH,
    # 'train_file': _PATH + 'data/train_data.json',
    # 'val_file': _PATH + 'data_pro/4_val_norm.json',
    
    # cls版本
    # 'val_file': _PATH + 'data/val_data_class.json',
    # 'test_file': _PATH + 'data/test_data_pred.json',
    # 'train_file': _PATH + 'data/train_data_class.json',
    
    'val_file': _PATH + 'meta/3_val_data_class.json',
    'test_file': _PATH + 'data/test_data_pred.json',
    'train_file': _PATH + 'meta/3_train_data_class.json',
    # 'train_file': _PATH + 'data_pro/4_train_norm.json',
    # 'train_file': _PATH + 'data_pro/3_normalized_data.json',
    'train_meta': _PATH + 'meta/3_train_meta.npy',
    'test_meta': _PATH + 'meta/3_test_meta.npy',
    'val_meta': _PATH + 'meta/3_val_meta.npy',
    'train_img': _PATH + 'data/01_data_train/',
    'test_img': _PATH + 'data/01_data_test/',
    'model': os.path.join(PROJECT_ROOT, 'model/'),
}

BASE_LEARNING_RATE = 1e-3
EPOCHS = 1
BATCH_SIZE = 1
WEIGHT_DECAY = 0.00005
