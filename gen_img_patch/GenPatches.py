import cv2
import numpy as np
import tensorflow as tf
from math import ceil
import argparse

def get_overlap(image_size, patch_size):
  from math import ceil
  n_overlap = image_size //patch_size
  n_patches = ceil(image_size /patch_size)
  return n_patches, (n_patches*patch_size - image_size) // n_overlap


  
def get_overlap_Npatches(image_size, patch_size, n_patches):
  n_overlap = n_patches - 1
  return n_patches, (n_patches*patch_size - image_size) // n_overlap


def get_start_points(image_size, patch_size):
  n_patches, overlap = get_overlap(image_size, patch_size)
  start_point = [0]*n_patches
  for i in range(1, n_patches):
    start_point[i] = start_point[i-1]+ patch_size - overlap
  return(start_point)

def get_start_points_Npatches(image_size, patch_size, n_patches):
  n_patches, overlap = get_overlap_Npatches(image_size, patch_size, n_patches)
  start_point = [0]*n_patches
  for i in range(1, n_patches):
    start_point[i] = start_point[i-1]+ patch_size - overlap
  return(start_point)



class get_patches(tf.keras.layers.Layer):

  def __init__( self , inputs, patch_size, n_patches_dim = None ):
    super( get_patches , self ).__init__()
    self.patch_size = patch_size
    self.len_x = inputs.shape[0]
    self.len_y = inputs.shape[1]
    self.n_patches = n_patches_dim
    range_i = get_start_points(image_size = self.len_x, patch_size = self.patch_size)  
    range_j = get_start_points(image_size = self.len_y, patch_size = self.patch_size)

    if n_patches_dim is not None:
      range_i = get_start_points_Npatches(image_size = self.len_x, patch_size = self.patch_size, n_patches = self.n_patches[0])  
      range_j = get_start_points_Npatches(image_size = self.len_y, patch_size = self.patch_size, n_patches = self.n_patches[1])

    patches = list() 
    position = list()

    for i in range_i:
        #row = row +1
        for j in range_j:
            #col = col + 1
            #print((row-1) + (col-1))
            patches.append(inputs[ i : i + self.patch_size , j : j + self.patch_size , : ])
            position.append([i+self.patch_size//2, j +self.patch_size//2])

    self.patches = patches
    self.position = position
    self.pairs = (len(range_i), len(range_j))

  def patches(self):
    return self.patches
  def position(self):
    return tf.cast(self.position)
  def paris(self):
    return self.pairs


if __name__ == "__main__":
    import tensorflow as tf
    print("this is main")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="image path")
    parser.add_argument("-npd", help="number of patches dimension, [r,c], total number=r*c. default gives you the least overlapping number of patches")
    parser.add_argument("-ps", help="patch size, defaults to 64")

    args = parser.parse_args()
    PATH = args.i
    N_PATCHES_DIM = args.npd
    if args.ps is None:
        patch_size = 64
    else: 
        patch_size = int(args.ps)

    file_name = PATH.split("/")[-1]
    raw = tf.io.read_file(PATH)
    img = tf.io.decode_jpeg(raw)
    res = get_patches(img, patch_size = patch_size, n_patches_dim = args.npd)
    print(res.position)
    
    i = 0
    for patch in res.patches:
        patch_jpeg = tf.image.encode_jpeg(patch, quality=100)
        patch_name = file_name.split(".")[0]+"__{}__{}.jpg".format(patch_size,i)
        print(patch_name)
        tf.io.write_file(patch_name, patch_jpeg)
        i = i +1
    
    