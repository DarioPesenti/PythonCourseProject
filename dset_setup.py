import os
import torch
print(torch.__version__)
from torch import nn # LP: unused import
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms # LP: unused import
from torch.utils.data import TensorDataset, DataLoader # LP: unused import
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE  # LP: unused import
from sklearn.decomposition import PCA  # LP: unused import
import pandas as pd  # LP: unused import
import numpy as np
import random  # LP: unused import
import h5py
import joblib
import argparse
from sklearn.model_selection import train_test_split




def sample_random_batch_with_labels(batch_size):
    """ Samples a random batch of images and their corresponding labels.
    Args:
        batch_size: number of images and labels to sample.

    Returns:
        images: shape [batch_size, 64, 64, 3], values normalized to range [0, 1].
        labels: shape [batch_size, number_of_label_dimensions]
    """
    # LP: I would recommend not to use variables out of scope for the function! maybe pass it as an argument?
    path =args.save_path

    dataset = h5py.File(path +'//3dshapes.h5', 'r')

    images = dataset['images']  # array shape [480000,64,64,3], uint8 in range(256)
    labels = dataset['labels']  # array shape [480000,6], float64
    image_shape = images.shape[1:]  # [64,64,3]  # LP: unused variable
    label_shape = labels.shape[1:]  # [6]
    n_samples = labels.shape[0]  # 10*10*10*8*4*15=480000   
    indices = np.random.choice(n_samples, batch_size)
    image_batch = []
    label_batch = []
    for ind in indices:
        image = images[ind]
        label = labels[ind]
        image = np.asarray(image, dtype=np.float32) / 255.  # normalise and convert to float32
        label_batch.append(label)
        image_batch.append(image)
    return np.stack(image_batch, axis=0), np.stack(label_batch, axis=0)

def show_images_grid(imgs_, num_images=25):
  ncols = int(np.ceil(num_images**0.5))
  nrows = int(np.ceil(num_images / ncols))
  _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
  axes = axes.flatten()

  for ax_i, ax in enumerate(axes):
    if ax_i < num_images:
      ax.imshow(imgs_[ax_i], cmap='Greys_r', interpolation='nearest')
      ax.set_xticks([])
      ax.set_yticks([])
    else:
      ax.axis('off')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--load', action='store_true', help='load previous dset')
    parser.add_argument('-ss','--sample_size', type=int, default=30000, help='set dset sample size to extract, default 30.000')
    parser.add_argument('--grid_images',type=int, help='show examples on grid, default = 0 thus no images shown', default=0)
    parser.add_argument('-path','--save_path', type=str, default=os.getcwd(), help='data save/load path')
    parser.add_argument('-s','--save', action='store_true', help='devide to save dataset or not (for debug)')
    args= parser.parse_args()
    if args.load:
        img_batch = np.load(args.save_path + '//img_batch.npy')
        label_batch = joblib.load(args.save_path + '//label_batch.pkl')
    else:
        img_batch, label_batch = sample_random_batch_with_labels(args.sample_size)
        if args.save:
            np.save(args.save_path + '\img_batch.npy', img_batch)
            joblib.dump(label_batch, args.save_path + '\label_batch.pkl')
    
    if args.grid_images!=0:
       show_images_grid(img_batch, args.grid_images)
       plt.show()



