import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import misc
import h5py

p = Path('.')
data = [item for item in p.glob('**/non_face/*.png')]

img = imread(data[5], mode='RGB')
img.shape

nonface_data = []
for item in data:  
    img = imread(item, mode='RGB')
    nonface_data.append(imresize(img, (224, 224)))

nonface_data[0].shape

h5f = h5py.File('nonface_test_data.h5', 'w')
h5f.create_dataset('X_train_1', data = nonface_data)