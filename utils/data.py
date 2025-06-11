import tensorflow as tf
import numpy as np
import pandas as pd
import os

def preprocess_dataset(file_path, property):
    def _process(file_path, property):
        image = np.load(file_path.numpy().decode())["data"]
        property = tf.cast(property, dtype=tf.float32)
        return image, property

    image, property = tf.py_function(_process, [file_path, property], [
        tf.float32, tf.float32])
    return image, property

def add_channel(image, property):
    image_with_channel = tf.expand_dims(image, axis=-1)
    return image_with_channel, property

def normalize_image(image):
    image = tf.cast(image, dtype=tf.float32)
    image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))
    return image

def prepare_dataset(dataset_path, properties_file, batch_size=32):
    properties_data = pd.read_csv(properties_file)
    Values = properties_data
    values_dataset = np.array([Values[i] for i in range(len(Values))])
    
    file_names = sorted(os.listdir(dataset_path), key=lambda x: int(x.split(".")[0]))
    files_path = [os.path.join(dataset_path, file_name) for file_name in file_names]
    
    dataset = tf.data.Dataset.from_tensor_slices((files_path, values_dataset))
    dataset = dataset.map(preprocess_dataset)
    dataset = dataset.map(add_channel)
    
    return dataset, len(files_path)
