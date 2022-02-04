
import os
import glob

from os.path import dirname, join
from PathConfig import *
from shutil import copy


root_path = dirname(os.path.realpath(__file__))
"""
    split train - test base on groundtruth

    train folder already clone from paris6k
"""

data_folder = FilePaths.train_set
data_class = os.listdir(data_folder)


# delete corrupt file 
with open(join(FilePaths.dataset, 'corrupt.txt'), 'r') as f:
    img_corrupts = [img[:-1] for img in f] # without '\n'
    
    for img in img_corrupts:
        os.remove(join(FilePaths.train_set, img))
        print('[REMOVE] {}'.format(img))

# copy image query to test_set
# test_list = []
# GT_folder = FilePaths.groundtruth
# for query_file in glob.glob(join(GT_folder, '*_query.txt')):
#     query = open(query_file, 'r')
#     query_image = query.read().split()[0] + '.jpg'
#     test_list.append(query_image)
#     copy(join(join(data_folder, query_image.split('_')[1]), query_image), FilePaths.test_set)

# remove image train from train_set
# img_list = ([os.listdir(FilePaths.dataset+x) for x in data_class])
# for img_class in data_class:
#     for img in os.listdir(join(data_folder, img_class)):
#         if img in test_list:
#             os.remove(join(join(data_folder, img_class), img))