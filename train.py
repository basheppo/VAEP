import tensorflow as tf
from models.vae import VAE
from utils.callbacks import CustomReduceLROnPlateau, CSVLogger
from utils.data import prepare_dataset
from tensorflow.keras.callbacks import ModelCheckpoint
from config import Config

def train_vae():
    dataset, dataset_size = prepare_dataset(
        Config.DATASET_PATH,
        Config.PROPERTIES_FILE,
        Config.BATCH_SIZE
    )
    
    train_size = int(0.9 * dataset_size)
    train_dataset = dataset.take(train_size).cache().batch(Config.BATCH_SIZE).prefetch(
        tf.data.experimental.AUTOTUNE)
    validation_dataset = dataset.skip(train_size).cache().batch(Config.BATCH_SIZE).prefetch(
        tf.data.experimental.AUTOTUNE)
    
    vae_model = VAE(latent_dim=Config.LATENT_DIM, input_shape=Config.INPUT_SHAPE)
    
    vae_adam = tf.keras.optimizers.Adam(
        learning_rate=Config.INITIAL_LEARNING_RATE, 
        clipnorm=1.0
    )
    
    vae_model.compile(vae_optimizer=vae_adam)
    
    callbacks = [
        ModelCheckpoint(
            filepath=Config.CHECKPOINT_PATH,
            save_weights_only=False,
            save_format="tf",
            monitor="total_loss",
            save_best_only=True,
            verbose=1
        ),
        CustomReduceLROnPlateau(
            monitor="val_total_loss",
            optimizer="vae_optimizer",
            factor=0.98,
            patience=5,
            min_lr=0.000001,
            verbose=1
        ),
        CSVLogger(Config.LOG_FILE)
    ]
    
    history = vae_model.fit(
        train_dataset,
        epochs=Config.EPOCHS,
        validation_data=validation_dataset,
        callbacks=callbacks
    )
    
    return history

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"Training on GPU: {gpus}")
    history = train_vae()