import tensorflow as tf
from tensorflow.keras.layers import Layer

class SamplingLayer(Layer):
    def call(self, inputs):
        mean, log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * epsilon


