"""
    Provides different backbone networks. 
    Includes:
        1. MobileNetv2
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2


class Backbone():
    def __init__(self, name="mobilenetv2", input_shape=(256, 256, 3), is_pretrained=True):
        self.name = name
        self.input_shape = input_shape 
        self.is_pretrained = is_pretrained
        self.feature_extractor = self.get_feature_extractor()

    def get_feature_extractor(self):
        if self.name == 'mobilenetv2':
            return self.get_mobilenetv2();

    def get_mobilenetv2(self):
        if self.is_pretrained:
            base_model = MobileNetV2(input_shape=self.input_shape, include_top=False, weights='imagenet')
        else:
            base_model = MobileNetV2(input_shape=self.input_shape, include_top=False)

        layer_names = [             # input: 256x256
            'block_1_expand_relu',  # 128x128 
            'block_3_expand_relu',  # 64x64
            'block_6_expand_relu',  # 32x32
            'block_13_expand_relu', # 16x16 
            'block_16_project'      # 8x8
        ]

        layers = [base_model.get_layer(name).output for name in layer_names]

        # create the feature extraction model
        feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=layers)
        feature_extractor.trainable = False
        return feature_extractor


