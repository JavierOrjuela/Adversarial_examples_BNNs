'''
VGG11/13/16/19 in TensorFlow2.

Reference:
[1] Simonyan, Karen, and Andrew Zisserman. 
    "Very deep convolutional networks for large-scale image recognition." 
    arXiv preprint arXiv:1409.1556 (2014).
'''
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
import sys
tfd = tfp.distributions
import tensorflow_probability as tfp

import functools

kernel_posterior_scale_constraint = 0.2
kernel_posterior_scale_mean = -9
kernel_posterior_scale_stddev = 0.1

def _untransformed_scale_constraint(t):
    return tf.clip_by_value(t, -1000,
                            tf.math.log(kernel_posterior_scale_constraint))

kernel_posterior_fn = tfp.layers.default_mean_field_normal_fn(
      untransformed_scale_initializer=tf.compat.v1.initializers.random_normal(
          mean=kernel_posterior_scale_mean,
          stddev=kernel_posterior_scale_stddev),
      untransformed_scale_constraint=_untransformed_scale_constraint)
divergence_fn = lambda q, p, _: tfd.kl_divergence(q, p) / 54000
## Divergencia

## Bayesian Convolutional Function

config = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    #'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg16': [16, 16, 'M', 32, 32, 'M', 64, 64, 64, 'M', 128, 128, 128 ,'M', 128, 128, 128, 'M'],
    #'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'vgg19': [16, 16, 'M', 32, 32, 'M', 64, 64, 64, 64, 'M', 128, 128, 128, 128, 'M', 128, 128, 128, 128, 'M'],
}

class VGG(tf.keras.Model):
    def __init__(self, vgg_name, num_classes):
        super(VGG, self).__init__()
        self.conv = self._make_layers(config[vgg_name])
        self.flatten = layers.Flatten()
        #self.fc = tfp.layers.DenseFlipout(num_classes)
        #self.fc = layers.Dense(num_classes)
        self.fc = tfp.layers.DenseReparameterization(num_classes)
        self.oh = tfp.layers.OneHotCategorical(num_classes, convert_to_tensor_fn=tfd.OneHotCategorical.mode)
        
    def call(self, x):
        out = self.conv(x)
        out = self.flatten(out)
        out = self.fc(out)
        out = self.oh(out)
        return out
    
    def _make_layers(self, config):
        layer = []
        for l in config:
            if l == 'M':
                layer += [layers.MaxPool2D(pool_size=2, strides=2)]
            else:
                layer += [#MNFConv2D(l, 3),
                          #layers.Conv2D(l, 3, padding = 'SAME'),
                          #tfp.layers.Convolution2DFlipout(l, 3,padding = 'SAME',kernel_posterior_fn=kernel_posterior_fn,kernel_divergence_fn = divergence_fn),
                          tfp.layers.Convolution2DReparameterization(l, 3, padding = 'SAME',kernel_posterior_fn=kernel_posterior_fn,kernel_divergence_fn = divergence_fn),
                          #layers.BatchNormalization(),
                          layers.ReLU()]
        layer += [layers.AveragePooling2D(pool_size=1, strides=1)]
        return tf.keras.Sequential(layer)