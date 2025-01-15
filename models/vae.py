import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv3D, Flatten, Conv3DTranspose, Reshape, LeakyReLU, ReLU, Dropout
from .layers import SamplingLayer, SteepSigmoid



class VAE(Model):
    def __init__(self, latent_dim=16, input_shape=(100, 100, 100, 1)):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape_ = input_shape
        self.beta = 10
        self.alpha = 100000
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.predictor = self.build_predictor()

    def set_myweights(self, new_model: Model, old_model: Model):
        old_weights = old_model.get_weights()
        new_model.set_weights(old_weights)

    def build_predictor(self):
        input = Input(shape=(self.latent_dim,), name="predictor_input_layer")
        x = Dense(units=64)(input)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dense(units=128)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dense(units=64)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dense(units=48)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dense(units=32)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dense(units=1, activation=None, dtype="float32",
                  name="predictor_output_layer")(x)
        x = ReLU()(x)
        predictor_output = x
        predictor = Model(input, predictor_output, name="predictor")
        return predictor

    def build_encoder(self):
        enc_in = Input(shape=self.input_shape_, name="input_layer_encoder")
        x = Conv3D(32, (3, 3, 3), strides=(2, 2, 2), padding="same")(
            enc_in)  # 32 32 64 64 128 128 256
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv3D(16, (3, 3, 3), strides=(1, 1, 1), padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv3D(32, (3, 3, 3), strides=(2, 2, 2), padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv3D(32, (4, 4, 4), strides=(1, 1, 1), padding="valid")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv3D(64, (4, 4, 4), strides=(2, 2, 2), padding="valid")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv3D(128, (3, 3, 3), strides=(2, 2, 2), padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Flatten()(x)
        x = Dense(2560, activation="relu")(x)
        x = Dropout(0.2)(x)
        mean = Dense(self.latent_dim, activation=None)(x)
        log_var = Dense(self.latent_dim, activation=None)(x)

        z = SamplingLayer()([mean, log_var])

        return Model(enc_in, [mean, log_var, z], name='encoder')

    def build_decoder(self):
        dec_in = Input(shape=(self.latent_dim,), name="input_layer_decoder")

        x = Dense(2560, activation='relu')(dec_in)
        x = Dropout(0.2)(x)
        x = Dense(5*5*5*128, activation='relu')(x)
        x = Reshape(target_shape=(5, 5, 5, 128))(x)
        x = Conv3DTranspose(128, (3, 3, 3), strides=(
            2, 2, 2), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv3DTranspose(64, (3, 3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv3DTranspose(64, (4, 4, 4), strides=(
            2, 2, 2), padding='valid')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv3DTranspose(32, (4, 4, 4), padding='valid')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv3DTranspose(32, (3, 3, 3), strides=(
            2, 2, 2), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv3DTranspose(16, (3, 3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv3DTranspose(16, (3, 3, 3), strides=(
            2, 2, 2), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv3DTranspose(1, (3, 3, 3), padding='same')(x)
        x = SteepSigmoid(k=5)(x)
        dec_out = x

        return Model(dec_in, dec_out, name='decoder')

    def compile(self, vae_optimizer):
        super().compile()
        self.vae_optimizer = vae_optimizer

    def call(self, inputs):
        images, properties = inputs
        mean, log_var, z = self.encoder(images)
        predicted_properties = self.predictor(mean)
        reconstructed = self.decoder(z)
        return reconstructed, mean, log_var, predicted_properties

    def compute_loss(self, data, reconstructed, mean, log_var, predicted_properties):
        images, properties = data
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(
                    images, reconstructed),
                axis=(1, 2, 3),
            )
        )

        predictor_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(properties - predicted_properties), axis=1))

        kl_loss = -0.5 * \
            tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
        vae_loss = reconstruction_loss + kl_loss
        total_loss = self.beta*reconstruction_loss + \
            kl_loss + self.alpha * predictor_loss

        return total_loss, reconstruction_loss, kl_loss, vae_loss, predictor_loss

    def val_compute_loss(self, images, reconstructed_images, mean, log_var, properties, predicted_properties):
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(
                    images, reconstructed_images),
                axis=(1, 2, 3),
            )
        )
        kl_loss = -0.5 * \
            tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))

        predictor_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(properties - predicted_properties), axis=1))

        return reconstruction_loss, kl_loss, predictor_loss

    @tf.function
    def train_step(self, data):
        with tf.GradientTape(persistent=True) as tape:
            reconstructed, mean, log_var, predicted_properties = self(
                data, training=True)
            total_loss, reconstruction_loss, kl_loss, vae_loss, predictor_loss = self.compute_loss(
                data, reconstructed, mean, log_var, predicted_properties)

        vaep_grads = tape.gradient(
            total_loss, self.encoder.trainable_variables+self.decoder.trainable_variables+self.predictor.trainable_variables)

        self.vae_optimizer.apply_gradients(
            zip(vaep_grads, self.encoder.trainable_variables+self.decoder.trainable_variables+self.predictor.trainable_variables))

        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "vae_loss": vae_loss,
            "predictor_loss": predictor_loss,
            "max_recon_value": tf.reduce_max(reconstructed),
            "min_recon_value": tf.reduce_min(reconstructed),
        }

    @tf.function
    def test_step(self, data):
        images, properties = data
        mean, log_var, z = self.encoder(images, training=False)
        reconstructed_images = self.decoder(z, training=False)
        predicted_properties = self.predictor(mean, training=False)
        val_reconstruction_loss, val_kl_loss, val_predictor_loss = self.val_compute_loss(
            images, reconstructed_images, mean, log_var, properties, predicted_properties
        )
        val_total_loss = val_kl_loss + val_reconstruction_loss + val_predictor_loss

        return {
            "reconstruction_loss": val_reconstruction_loss,
            "kl_loss": val_kl_loss,
            "predictor_loss": val_predictor_loss,
            "total_loss": val_total_loss
        }