import tensorflow as tf
from tensorflow.keras.layers import Layer

class SamplingLayer(Layer):
    def call(self, inputs):
        mean, log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * epsilon

class SteepSigmoid(Layer):
    def __init__(self, k=5, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.k = k

    def call(self, inputs):
        return self.steep_sigmoid(inputs)

    def steep_sigmoid(self, inputs):
        inputs = tf.clip_by_value(inputs, -10, 10)
        return 1 / (1+tf.exp(-self.k*inputs))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"k": self.k}
        base_config = super().get_config()
        return {**base_config, **config}
