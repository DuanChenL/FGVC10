"""
将原有的csv正则化存为新的csv
"""

import numpy as np
import pandas as pd
import json

# define path
_PATH = r'../'

# file path
train_meta_path = _PATH + 'data/data_train_mean.csv'

# read data
# train_df = pd.read_csv(train_meta_path,low_memory=False,usecols=[19,20,26,30,37,38])
train_df = pd.read_csv(train_meta_path,low_memory=False,usecols=[19,20,30,37,38])

# train_df = pd.read_csv(train_meta_path,low_memory=False)

# for postProcess
# lst = []
# global mean_4,mean_18,mean_46,mean_144,mean_145,std_4,std_18,std_46,std_144,std_145
data = {
    "mean_4": train_df['trait_4'].mean(),
    "mean_46": train_df['trait_46'].mean(),
    # "mean_144": train_df['trait_144'].mean(),
    # "mean_145": train_df['trait_145'].mean(),
    "std_4": train_df['trait_4'].std(),
    "std_46": train_df['trait_46'].std(),
    # "std_144": train_df['trait_144'].std(),
    # "std_145": train_df['trait_145'].std()
}
with open("../data/000_rec_train_mean_std_trait_4.json", 'w') as file:
    # lst.append(data)
    # json.dump(list, file,indent=1)
    json.dump(data,file)


# rec_df = pd.DataFrame()
# rec_df['mean_4'] = train_df['trait_4'].mean()
# rec_df['mean_18'] = train_df['trait_18'].mean()
# rec_df['mean_46'] = train_df['trait_46'].mean()
# rec_df['mean_144'] = train_df['trait_144'].mean()
# rec_df['mean_145'] = train_df['trait_145'].mean()
# rec_df['std_4'] = train_df['trait_4'].std()
# rec_df['std_18'] = train_df['trait_4'].std()
# rec_df['std_46'] = train_df['trait_46'].std()
# rec_df['std_144'] = train_df['trait_144'].std()
# rec_df['std_145'] = train_df['trait_145'].std()
# print(train_df['trait_4'].mean())

# normalization 标准正则
train_df['trait_4'] = (train_df['trait_4']-train_df['trait_4'].mean())/train_df['trait_4'].std()
# train_df['trait_18'] = (train_df['trait_18']-train_df['trait_18'].mean())/train_df['trait_18'].std()
train_df['trait_46'] = (train_df['trait_46']-train_df['trait_46'].mean())/train_df['trait_46'].std()
train_df['trait_144'] = (train_df['trait_144']-train_df['trait_144'].mean())/train_df['trait_144'].std()
train_df['trait_145'] = (train_df['trait_145']-train_df['trait_145'].mean())/train_df['trait_145'].std()

# save
train_df.to_csv(_PATH + 'data/001_train_mean_trait_4.csv', index=False) #仅仅只正则化4 to 15
# rec_df.to_csv(_PATH + 'data/000_rec_train_mean_std_trait_5.csv', index=False) #仅仅只正则化4 to 15