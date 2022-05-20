### Patch Images using TensorFlow
Generating Patches from Given Images
Set number of patches and patch sizes, which will result in overlapping patches. 

![Alt text](data/gen_image_patch_example.png?raw=true)


```python

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


```