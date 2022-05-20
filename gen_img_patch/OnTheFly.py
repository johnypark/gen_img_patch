__copyright__ = """
Copyright (c) 2022 John Park
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions: The above copyright notice and this permission
notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

import tensorflow as tf


def gen_patch_from_batch(image, label, patch_size, n_batch, n_patches = None):
  """ generate patches on the fly from batch in training time.
  This function can convert (32, 480, 480, 3) tensor to (128, 160, 160, 3) that of 
  any patch sizes 
  Currently only works for square image and square patches. 
  Must specify n_batch and n_patches: number of patches in one axis.
  Usage example:

  import tensorflow_datasets as tfds

  data, ds_info = tfds.load("oxford_flowers102", with_info=True, as_supervised=True)
  train_ds = data["train"]
  num_classes = ds_info.features["label"].num_classes
  n_batch = 32
  imsize = (448, 448)
  train_ds = data["train"]
  train_ds = train_ds.map(
      lambda image, label: 
      (tf.image.resize(image, imsize), tf.one_hot(label, num_classes)))
  train_ds = train_ds.batch(n_batch)

  train_ds = train_ds.map(
      lambda image, label: 
      gen_patch_from_batch(image, 
                            label, 
                            n_batch = 32, 
                            patch_size = 224, 
                            n_patches = 4))
  """
  def get_overlap(image_size, patch_size, n_patches = None):
    from math import ceil
    if n_patches is None:
      n_overlap = image_size //patch_size
      n_patches = ceil(image_size /patch_size)
    else:
      n_overlap = n_patches - 1
    return n_patches, (n_patches*patch_size - image_size) // n_overlap

  def crop_batch(batch, n_patch, patch_size):
    import math
    remain = batch.shape[1] - n_patch * patch_size 
    if remain > 0:
      margin = math.ceil(remain/2)
      #print(margin)
      max_pos = batch.shape[1] - margin
      #print(max_pos)
      min_pos = margin
      #print(min_pos)
      batch = batch[:,min_pos: max_pos, min_pos: max_pos, :]
    return batch

  print(image.shape)
  image = crop_batch(image, n_patch = n_patches, patch_size = patch_size)
  image_size = image.shape[1]
 

  print(image_size)
  print(n_batch)
  n_patches, overlap = get_overlap(image_size = image_size, patch_size = patch_size, 
                     n_patches = n_patches)
  print(n_patches, overlap)
  result = tf.image.extract_patches(images = image,
                           sizes=[1, patch_size, patch_size, 1],
                           strides=[1, (patch_size - overlap), (patch_size -overlap), 1],
                           rates=[1, 1, 1, 1],
                           padding='VALID')
  print(result.shape[1])
  #print([n_batch*(result.shape[1])*(result.shape[2]), patch_size, patch_size, 3])
  result_reshape = tf.reshape(result, [n_batch*(result.shape[1])*(result.shape[2]), patch_size, patch_size, 3])
  actual_n_patch = result_reshape.shape[0]/n_batch
  update_label = tf.repeat(label, int(actual_n_patch), axis = 0)
  return result_reshape, update_label