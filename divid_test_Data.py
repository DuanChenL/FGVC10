import os
from PIL import Image
import random
import json
import numpy as np

'''
这个文件名可能有点干扰性，这个文件用的是train版本的逻辑，处理json为npy文件，跳过了proprocess.py
'''

if __name__ == '__main__':
    path = './meta'
    all_img = 0
    test_meta = []
    with open(path + '/1_test_data_class.json', 'r') as file:
        file = json.load(file)
        # print(file)
        for _data in file:
            # print(_data)
            temp_meta = []
            temp_meta = [float(x) for x in _data['bio']]
            test_meta.append(temp_meta)
                   
    a=np.array(test_meta)
    print(a)
    print(a.shape)
    np.save('meta/3_test_meta.npy',a)   # 保存为.npy格式
    