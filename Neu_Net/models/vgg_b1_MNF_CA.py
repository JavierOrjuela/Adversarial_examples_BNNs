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
import sys
sys.path.insert(1, 'models/tf-mnf')
import tensorflow_probability as tfp
from tf_mnf.layers import MNFConv2D, MNFDense

import functools
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
        self.num_classes=num_classes
        self.conv = self._make_layers(config[vgg_name])
        self.flatten = layers.Flatten()
        self.fc = MNFDense(num_classes, prior_choice='standard_cauchy')
        self.oh = tfp.layers.OneHotCategorical(num_classes)
        
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
                layer += [MNFConv2D(l, 3, prior_choice='standard_cauchy'), 
                          layers.BatchNormalization(),
                          layers.ReLU()]
        layer += [layers.AveragePooling2D(pool_size=1, strides=1)]
        return tf.keras.Sequential(layer)
    
    def kl_div(self):
        """Compute current KL divergence of the whole model.
        Can be used as a regularization term during training.
        """
        def rgetattr(obj, attr):
            def _getattr(obj, attr):
                return hasattr(obj, attr)
            return functools.reduce(_getattr, [obj] + attr.split('.'))
        #import pdb;pdb.set_trace()
        return sum(lyr.kl_div() for lyr in self.layers if rgetattr(lyr, "kl_div"))+sum(lyr.kl_div() for lyr in self.layers[0].layers  if rgetattr(lyr, "kl_div"))
    
    
    
    