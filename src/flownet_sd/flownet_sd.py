from ..net import Net, Mode
from ..utils import LeakyReLU, average_endpoint_error, pad, antipad
from ..downsample import downsample
import math
import tensorflow as tf
from ..training_schedules import LONG_SCHEDULE
slim = tf.contrib.slim


class FlowNetSD(Net):

    def __init__(self, mode=Mode.TRAIN, debug=False):
        super(FlowNetSD, self).__init__(mode=mode, debug=debug)
        self.model(None, LONG_SCHEDULE)


    def model(self, inputs, training_schedule, trainable=True):
        w = 1152#512
        h = 320#384
        c = 3
        self.input_a = tf.placeholder(tf.float32, [h, w, c])
        self.input_b = tf.placeholder(tf.float32, [h, w, c])
        self.X1 = tf.expand_dims(self.input_a, 0)
        self.X2 = tf.expand_dims(self.input_b, 0)
        self.X = tf.concat([self.X1, self.X2], axis=3)

        print (self.X.get_shape())

        with tf.variable_scope('FlowNetSD'):
            print ("im here 1 ----------------------------------------------------")

            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                # Only backprop this network if trainable
                                trainable=trainable,
                                # He (aka MSRA) weight initialization
                                weights_initializer=slim.variance_scaling_initializer(),
                                activation_fn=LeakyReLU,
                                # We will do our own padding to match the original Caffe code
                                padding='VALID'):
                print ("im here 2 ----------------------------------------------------")
                weights_regularizer = slim.l2_regularizer(training_schedule['weight_decay'])
                with slim.arg_scope([slim.conv2d], weights_regularizer=weights_regularizer):
                    conv0 = slim.conv2d(pad(self.X), 64, 3, scope='conv0')
                    conv1 = slim.conv2d(pad(conv0), 64, 3, stride=2, scope='conv1')
                    conv1_1 = slim.conv2d(pad(conv1), 128, 3, scope='conv1_1')
                    conv2 = slim.conv2d(pad(conv1_1), 128, 3, stride=2, scope='conv2')
                    conv2_1 = slim.conv2d(pad(conv2), 128, 3, scope='conv2_1')
                    conv3 = slim.conv2d(pad(conv2_1), 256, 3, stride=2, scope='conv3')
                    conv3_1 = slim.conv2d(pad(conv3), 256, 3, scope='conv3_1')
                    conv4 = slim.conv2d(pad(conv3_1), 512, 3, stride=2, scope='conv4')
                    conv4_1 = slim.conv2d(pad(conv4), 512, 3, scope='conv4_1')
                    conv5 = slim.conv2d(pad(conv4_1), 512, 3, stride=2, scope='conv5')
                    conv5_1 = slim.conv2d(pad(conv5), 512, 3, scope='conv5_1')
                    conv6 = slim.conv2d(pad(conv5_1), 1024, 3, stride=2, scope='conv6')
                    conv6_1 = slim.conv2d(pad(conv6), 1024, 3, scope='conv6_1')
                    print ("im here 4 ----------------------------------------------------")
                    print(conv6_1.get_shape())

                    """ START: Refinement Network """
                    with slim.arg_scope([slim.conv2d_transpose], biases_initializer=None):
                        deconv5 = antipad(slim.conv2d_transpose(conv6_1, 512, 4, stride=2, scope='deconv5'))
                        predict_flow6 = slim.conv2d(pad(conv6_1), 2, 3, scope='predict_flow6', activation_fn=None)
                        #predict_flow6 = slim.conv2d(pad(conv6_1), 2, 3, scope='predict_flow6', activation_fn=LeakyReLU)
                        upsample_flow6to5 = antipad(slim.conv2d_transpose(predict_flow6, 2, 4, stride=2, scope='upsample_flow6to5', activation_fn=None))
                        #upsample_flow6to5 = antipad(slim.conv2d_transpose(predict_flow6, 2, 4, stride=2, scope='upsample_flow6to5', activation_fn=LeakyReLU))
                        print ("im here 4_1 ----------------------------------------------------")
                        print(conv5_1.get_shape())
                        print(deconv5.get_shape())
                        print(upsample_flow6to5.get_shape())
                        concat5 = tf.concat([conv5_1, deconv5, upsample_flow6to5], axis=3)


                        interconv5 = slim.conv2d(pad(concat5), 512, 3, activation_fn=None, scope='interconv5')
                        #interconv5 = slim.conv2d(pad(concat5), 512, 3, activation_fn=LeakyReLU, scope='interconv5')

                        predict_flow5 = slim.conv2d(pad(interconv5), 2, 3, scope='predict_flow5', activation_fn=None)
                        #predict_flow5 = slim.conv2d(pad(interconv5), 2, 3, scope='predict_flow5', activation_fn=LeakyReLU)
                        deconv4 = antipad(slim.conv2d_transpose(concat5, 256, 4, stride=2, scope='deconv4'))
                        upsample_flow5to4 = antipad(slim.conv2d_transpose(predict_flow5, 2, 4, stride=2, scope='upsample_flow5to4', activation_fn=None))
                        #upsample_flow5to4 = antipad(slim.conv2d_transpose(predict_flow5, 2, 4, stride=2, scope='upsample_flow5to4', activation_fn=LeakyReLU))
                        concat4 = tf.concat([conv4_1, deconv4, upsample_flow5to4], axis=3)
                        print ("im here 4_2 ----------------------------------------------------")
                        print(interconv5.get_shape())
                        print(predict_flow5.get_shape())
                        print(deconv4.get_shape())
                        print(upsample_flow5to4.get_shape())
                        print(concat4.get_shape())


                        interconv4 = slim.conv2d(pad(concat4), 256, 3, activation_fn=None, scope='interconv4')

                        predict_flow4 = slim.conv2d(pad(interconv4), 2, 3, scope='predict_flow4', activation_fn=None)
                        deconv3 = antipad(slim.conv2d_transpose(concat4, 128, 4, stride=2, scope='deconv3'))
                        upsample_flow4to3 = antipad(slim.conv2d_transpose(predict_flow4, 2, 4,
                                                                          stride=2,
                                                                          scope='upsample_flow4to3',
                                                                          activation_fn=None))
                        print ("im here 4_3 ----------------------------------------------------")
                        print(interconv4.get_shape())
                        print(predict_flow4.get_shape())
                        print(deconv3.get_shape())
                        print(upsample_flow4to3.get_shape())

                        print ("im here 5 ----------------------------------------------------")
                        print(conv3_1.get_shape())
                        print(deconv3.get_shape())
                        print(upsample_flow4to3.get_shape())
                        concat3 = tf.concat([conv3_1, deconv3, upsample_flow4to3], axis=3)
                        interconv3 = slim.conv2d(pad(concat3), 128, 3, activation_fn=None, scope='interconv3')

                        predict_flow3 = slim.conv2d(pad(interconv3), 2, 3, scope='predict_flow3', activation_fn=None)
                        deconv2 = antipad(slim.conv2d_transpose(concat3, 64, 4, stride=2, scope='deconv2'))
                        upsample_flow3to2 = antipad(slim.conv2d_transpose(predict_flow3, 2, 4,
                                                                          stride=2,
                                                                          scope='upsample_flow3to2',
                                                                          activation_fn=None))
                        print ("im here 5-2 ----------------------------------------------------")
                        print(interconv3.get_shape())
                        print(predict_flow3.get_shape())
                        print(deconv2.get_shape())
                        print(upsample_flow3to2.get_shape())
                        print(concat4.get_shape())


                        concat2 = tf.concat([conv2, deconv2, upsample_flow3to2], axis=3)

                        print ("im here 6 ----------------------------------------------------")
                        interconv2 = slim.conv2d(pad(concat2), 64, 3, activation_fn=None, scope='interconv2')

                        predict_flow2 = slim.conv2d(pad(interconv2), 2, 3, scope='predict_flow2', activation_fn=None)


                        print ("im here 6 ----------------------------------------------------")
                        print(conv2.get_shape())
                        print(deconv2.get_shape())
                        print(upsample_flow3to2.get_shape())
                    """ END: Refinement Network """

                    flow = predict_flow2 * 0.05
                    # TODO: Look at Accum (train) or Resample (deploy) to see if we need to do something different
                    print ("im here 7")
                    print (flow.get_shape())
                    flow = tf.image.resize_bilinear(flow, tf.stack([h, w]), align_corners=True)
                    print (flow.get_shape())
                    convOut = conv6_1[0,:,:,:]
                    #convOut = tf.contrib.layers.flatten(convOut)
                    convOut = tf.reshape(convOut, [-1])
                    self.result = {
                        'predict_flow6': predict_flow6,
                        'predict_flow5': predict_flow5,
                        'predict_flow4': predict_flow4,
                        'predict_flow3': predict_flow3,
                        'predict_flow2': predict_flow2,
                        'flow': flow,
                        'convOut': convOut,
                    }


    def loss(self, flow, predictions):
        flow = flow * 20.0

        losses = []
        INPUT_HEIGHT, INPUT_WIDTH = float(flow.shape[1].value), float(flow.shape[2].value)

        # L2 loss between predict_flow6, blob23 (weighted w/ 0.32)
        predict_flow6 = predictions['predict_flow6']
        size = [predict_flow6.shape[1], predict_flow6.shape[2]]
        downsampled_flow6 = downsample(flow, size)
        losses.append(average_endpoint_error(downsampled_flow6, predict_flow6))

        # L2 loss between predict_flow5, blob28 (weighted w/ 0.08)
        predict_flow5 = predictions['predict_flow5']
        size = [predict_flow5.shape[1], predict_flow5.shape[2]]
        downsampled_flow5 = downsample(flow, size)
        losses.append(average_endpoint_error(downsampled_flow5, predict_flow5))

        # L2 loss between predict_flow4, blob33 (weighted w/ 0.02)
        predict_flow4 = predictions['predict_flow4']
        size = [predict_flow4.shape[1], predict_flow4.shape[2]]
        downsampled_flow4 = downsample(flow, size)
        losses.append(average_endpoint_error(downsampled_flow4, predict_flow4))

        # L2 loss between predict_flow3, blob38 (weighted w/ 0.01)
        predict_flow3 = predictions['predict_flow3']
        size = [predict_flow3.shape[1], predict_flow3.shape[2]]
        downsampled_flow3 = downsample(flow, size)
        losses.append(average_endpoint_error(downsampled_flow3, predict_flow3))

        # L2 loss between predict_flow2, blob43 (weighted w/ 0.005)
        predict_flow2 = predictions['predict_flow2']
        size = [predict_flow2.shape[1], predict_flow2.shape[2]]
        downsampled_flow2 = downsample(flow, size)
        losses.append(average_endpoint_error(downsampled_flow2, predict_flow2))

        loss = tf.losses.compute_weighted_loss(losses, [0.32, 0.08, 0.02, 0.01, 0.005])

        # Return the 'total' loss: loss fns + regularization terms defined in the model
        return tf.losses.get_total_loss()
