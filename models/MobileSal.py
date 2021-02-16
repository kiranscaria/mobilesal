"""
    Implementation of the Efficient MobileSal architecture
"""

import os
import sys
import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Conv2D, ReLU, \
    BatchNormalization, Dense, Activation, DepthwiseConv2D, Conv2DTranspose, Add, \
    Multiply, Concatenate, Reshape

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

from models.Backbone import Backbone


def conv1x1(x_in, out_filters, use_sigmoid_activation=False, name=''):
    x = Conv2D(filters=out_filters, kernel_size=1, name=f'{name}_conv')(x_in)
    x = BatchNormalization(axis=-1, name=f'{name}_bn')(x)
    if use_sigmoid_activation:
        x = Activation('sigmoid', name=f'{name}_act')(x)
    else:
        x = ReLU(name=f'{name}_relu')(x)
    return x


def conv3x3(x_in, out_filters, strides=1, name=''):
    x = Conv2D(filters=out_filters, kernel_size=3, strides=strides, padding="SAME",
               use_bias=False, kernel_initializer='he_normal', name=f'{name}_conv')(x_in)
    x = BatchNormalization(axis=-1, name=f'{name}_bn')(x)
    x = ReLU(name=f'{name}_relu')(x)
    return x


def upsamplex2(x_in, out_filters, name=''):
    x = Conv2DTranspose(filters=out_filters, kernel_size=3, strides=2, padding="SAME",
                        use_bias=False, kernel_initializer='he_normal', name=f'{name}_upx2')(x_in)
    x = BatchNormalization(axis=-1, name=f'{name}_bn')(x)
    x = ReLU(name=f'{name}_relu')(x)
    return x


# TODO: Need to check if the inverted residual block is working
def inverted_residual_block(x_in, expand, squeeze, strides=1, name=''):
    x = Conv2D(expand, kernel_size=1, name=f'{name}_squeeze_conv')(x_in)
    x = BatchNormalization(axis=-1, name=f'{name}_squeeze_bn')(x)
    x = ReLU(name=f'{name}_squeeze_relu')(x)
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='SAME',
                        name=f'{name}_dwconv')(x)
    x = BatchNormalization(axis=-1, name=f'{name}_dw_bn')(x)
    x = ReLU(name=f'{name}_dw_relu')(x)
    x = Conv2D(squeeze, kernel_size=1, name=f'{name}_expand_conv')(x)
    x = BatchNormalization(axis=-1, name=f'{name}_expand_bn')(x)

    if strides == 1:
        x = Add(name=f'{name}_add')([x, x_in])

    return x


def implicit_depth_restoration(d_in, out_channels=256, width_multiplier=4):
    # making the output channels equal
    d1 = Conv2D(filters=out_channels, kernel_size=1,
                name='idr_conv1x1_d1')(d_in[0])
    d2 = Conv2D(filters=out_channels, kernel_size=1,
                name='idr_conv1x1_d2')(d_in[1])
    d3 = Conv2D(filters=out_channels, kernel_size=1,
                name='idr_conv1x1_d3')(d_in[2])
    d4 = Conv2D(filters=out_channels, kernel_size=1,
                name='idr_conv1x1_d4')(d_in[3])
    d5 = Conv2D(filters=out_channels, kernel_size=1,
                name='idr_conv1x1_d5')(d_in[4])

    # making the output feature-maps equal
    d1 = conv3x3(d1, out_filters=out_channels, strides=2, name="idr_d1_0")
    d1 = conv3x3(d1, out_filters=out_channels, strides=2, name="idr_d1_1")
    d2 = conv3x3(d2, out_filters=out_channels, strides=2, name="idr_d2_0")
    d4 = upsamplex2(d4, out_filters=out_channels, name="idr_d4_0")
    d5 = upsamplex2(d5, out_filters=out_channels, name="idr_d5_0")
    d5 = upsamplex2(d5, out_filters=out_channels, name="idr_d5_1")

    # combining everything
    x = Concatenate(axis=-1, name='idr_concat')([d1, d2, d3, d4, d5])
    x = conv1x1(x, out_filters=256, name='idr_shrink')
    x = inverted_residual_block(
        x_in=x, expand=256*width_multiplier, squeeze=256, name='idr_irb_0')
    x = inverted_residual_block(
        x_in=x, expand=256*width_multiplier, squeeze=256, name='idr_irb_1')
    x = inverted_residual_block(
        x_in=x, expand=256*width_multiplier, squeeze=256, name='idr_irb_2')
    x = inverted_residual_block(
        x_in=x, expand=256*width_multiplier, squeeze=256, name='idr_irb_3')
    x = conv1x1(x, out_filters=1, name='idr_conv1x1_final')
    return x


def compact_pyramid_refinement(x_in, expand, squeeze, name=''):
    in_channels = x_in.shape[-1]
    x = conv1x1(x_in, out_filters=expand, name=f'{name}_conv1x1_0')
    x_d1 = DepthwiseConv2D(kernel_size=3, dilation_rate=1, padding="SAME", use_bias=False,
                           kernel_initializer='he_normal', name=f'{name}_dwconv0')(x)
    x_d2 = DepthwiseConv2D(kernel_size=3, dilation_rate=2, padding="SAME", use_bias=False,
                           kernel_initializer='he_normal', name=f'{name}_dwconv2')(x)
    x_d3 = DepthwiseConv2D(kernel_size=3, dilation_rate=3, padding="SAME", use_bias=False,
                           kernel_initializer='he_normal', name=f'{name}_dwconv3')(x)
    x_right = Add(name=f'{name}_right_add')([x_d1, x_d2, x_d3])
    x_right = BatchNormalization(axis=-1, name=f'{name}_right_bn0')(x_right)
    x_right = ReLU(name=f'{name}_right_relu0')(x_right)

    x_right = conv1x1(x_right, out_filters=in_channels, name=f'{name}_conv1x1_1')
    x_right = Add(name=f'{name}_combine_add')([x_right, x_in])
    x_right = conv1x1(x_right, out_filters=squeeze, name=f'{name}_combine')

    x_left = GlobalAveragePooling2D(name=f'{name}_GAP')(x_in)
    x_left = Reshape(target_shape=(1, 1, -1), name=f'{name}_gap_reshape')(x_left)
    x_left = Conv2D(filters=squeeze, kernel_size=1,
                    name=f'{name}_conv_final0')(x_left)
    x_left = BatchNormalization(axis=-1, name=f'{name}_bn_final0')(x_left)
    x_left = ReLU(name=f'{name}_relu_final0')(x_left)
    x_left = Conv2D(filters=squeeze, kernel_size=1,
                    name=f'{name}_conv_final1')(x_left)
    x_left = BatchNormalization(axis=-1, name=f'{name}_bn_final1')(x_left)
    x_left = Activation('sigmoid', name=f'{name}_act_final1')(x_left)

    x = Multiply()([x_right, x_left])
    return x


def cross_modality_fusion(image_feature_in, depth_feature_in, backbone_out_channels, width_multiplier):
    combined = Multiply(name='cmf_multiply0')(
        [image_feature_in, depth_feature_in])
    combined = inverted_residual_block(combined, expand=backbone_out_channels*width_multiplier,
                                       squeeze=backbone_out_channels, name="cmf_irb_0")

    x_image = GlobalAveragePooling2D(name="cmf_gap")(image_feature_in)
    x_image = Dense(units=backbone_out_channels, name="cmf_dense0")(x_image)
    x_image = BatchNormalization(name='cmf_bn0')(x_image)
    x_image = ReLU(name='cmf_relu0')(x_image)
    x_image = Dense(units=backbone_out_channels, name='cmf_relu1')(x_image)
    x_image = Activation('sigmoid', name='cmf_act1')(x_image)

    x_combined = Multiply(name='cmf_multiply1')([x_image, combined])
    x_combined = inverted_residual_block(x_combined, expand=backbone_out_channels*width_multiplier,
                                         squeeze=backbone_out_channels,
                                         name='cmf_irb_final')
    return x_combined


def depth_path(depth_in, out_filters=[16, 32, 64, 96], final_filters=320, width_multiplier=4):
    d1 = inverted_residual_block(depth_in, expand=out_filters[0]*width_multiplier,
                                 squeeze=out_filters[0], strides=2, name='depth_l0_irb0')
    d1 = inverted_residual_block(d1, expand=out_filters[0]*width_multiplier,
                                 squeeze=out_filters[0], strides=1, name='depth_l0_irb1')

    d2 = inverted_residual_block(d1, expand=out_filters[1]*width_multiplier,
                                 squeeze=out_filters[1], strides=2, name='depth_l1_irb0')
    d2 = inverted_residual_block(d2, expand=out_filters[1]*width_multiplier,
                                 squeeze=out_filters[1], strides=1, name='depth_l1_irb1')

    d3 = inverted_residual_block(d2, expand=out_filters[2]*width_multiplier,
                                 squeeze=out_filters[2], strides=2, name='depth_l2_irb0')
    d3 = inverted_residual_block(d3, expand=out_filters[2]*width_multiplier,
                                 squeeze=out_filters[2], strides=1, name='depth_l2_irb1')

    d4 = inverted_residual_block(d3, expand=out_filters[3]*width_multiplier,
                                 squeeze=out_filters[3], strides=2, name='depth_l3_irb0')
    d4 = inverted_residual_block(d4, expand=out_filters[3]*width_multiplier,
                                 squeeze=out_filters[3], strides=1, name='depth_l3_irb1')

    d5 = inverted_residual_block(d4, expand=final_filters*width_multiplier,
                                 squeeze=final_filters, strides=2, name='depth_l4_irb0')
    d5 = inverted_residual_block(d5, expand=final_filters*width_multiplier,
                                 squeeze=final_filters, strides=1, name='depth_l4_irb1')

    return d5


def mobilesal(input_shape=(224, 224, 3), backbone='mobilenetv2', is_training=True):
    image_in = Input(shape=input_shape, name='image_in')
    depth_in = Input(shape=(
        input_shape[0], input_shape[1], 1), name='depth_in')

    backbone = Backbone(
        name=backbone, input_shape=input_shape, is_pretrained=True)
    x1, x2, x3, x4, x5 = backbone.get_feature_extractor()(image_in)

    backbone_out_channels = x5.shape[-1]

    d_out = depth_path(depth_in, out_filters=[16, 32, 64, 96],
                       final_filters=backbone_out_channels, width_multiplier=4)

    cmf_out = cross_modality_fusion(image_feature_in=x5, depth_feature_in=d_out,
                                    backbone_out_channels=backbone_out_channels,
                                    width_multiplier=4)

    # Implicit Depth Restoration (IDR)
    x_idf = implicit_depth_restoration([x1, x2, x3, x4, cmf_out])

    # Decoder
    x_cpr5 = compact_pyramid_refinement(cmf_out, expand=backbone_out_channels*4,
                                        squeeze=backbone_out_channels, name="cpr5")

    x_cpr4 = upsamplex2(x_cpr5, backbone_out_channels, name="cpr5_x2")
    x_cpr4 = Concatenate(axis=-1)([x_cpr4, x4])
    x_cpr4 = compact_pyramid_refinement(x_cpr4, expand=96*4, squeeze=96, name="cpr4")

    x_cpr3 = upsamplex2(x_cpr4, 96, name="cpr4_x2")
    x_cpr3 = Concatenate(axis=-1)([x_cpr3, x3])
    x_cpr3 = compact_pyramid_refinement(x_cpr3, expand=64*4, squeeze=64, name="cpr3")

    x_cpr2 = upsamplex2(x_cpr3, 64, name="cpr3_x2")
    x_cpr2 = Concatenate(axis=-1)([x_cpr2, x2])
    x_cpr2 = compact_pyramid_refinement(x_cpr2, expand=32*4, squeeze=32, name="cpr2")

    x_cpr1 = upsamplex2(x_cpr2, 32, name="cpr2_x2")
    x_cpr1 = Concatenate(axis=-1)([x_cpr2, x2])
    x_cpr1 = compact_pyramid_refinement(x_cpr1, expand=16*4, squeeze=16, name="cpr1")

    x_cpr5_out = conv1x1(x_cpr5, out_filters=1, use_sigmoid_activation=True, name="cpr5_out")
    x_cpr4_out = conv1x1(x_cpr4, out_filters=1, use_sigmoid_activation=True, name="cpr4_out")
    x_cpr3_out = conv1x1(x_cpr3, out_filters=1, use_sigmoid_activation=True, name="cpr3_out")
    x_cpr2_out = conv1x1(x_cpr2, out_filters=1, use_sigmoid_activation=True, name="cpr2_out")
    x_cpr1_out = conv1x1(x_cpr1, out_filters=1, use_sigmoid_activation=True, name="cpr1_out")

    if is_training:
        return tf.keras.models.Model(inputs=[image_in, depth_in],
                                     outputs=[x_idf, [x_cpr1_out, x_cpr2_out, x_cpr3_out, x_cpr4_out, x_cpr5_out]])
    else:
        return tf.keras.models.Model(inputs=[image_in, depth_in], outputs=x_cpr1_out)


def test_model():
    model = mobilesal(input_shape=(224, 224, 3),
                      backbone='mobilenetv2', is_training=True)
    print(model.summary())


if __name__ == '__main__':
    test_model()
