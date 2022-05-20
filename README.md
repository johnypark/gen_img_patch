### Patch Images using TensorFlow

Patch

A patch (also called a local surface) is a differentiable mapping x:U->R^n, where U is an open subset of R^2. More generally, if A is any subset of R^2, then a map x:A->R^n is a patch provided that x can be extended to a differentiable map from U into R^n, where U is an open set containing A. Here, x(U) (or more generally, x(A)) is called the map trace of x.

from https://mathworld.wolfram.com/Patch.html

Guess what this flower's name is!

![Alt text](data/gen_image_patch_example.png?raw=true)


Usage example:

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