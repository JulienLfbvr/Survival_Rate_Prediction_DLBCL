import pathlib
import pickle
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras import Input
from keras import backend
from keras.applications import DenseNet121
from keras.applications.densenet import conv_block
from tensorflow.keras import layers
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

patch_dir = "D:\\ISEN\\M1\\Projet M1\\DLBCL-Morph\\Patches\\HE"
network_weights_address = "../weights/KimiaNetKerasWeights.h5"
network_input_patch_width = 224
img_format = 'png'
AUTOTUNE = tf.data.experimental.AUTOTUNE


# =============================================================


def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_png(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, [network_input_patch_width, network_input_patch_width])


def process_path(file_path):
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, img


def data_augmentation(batch, blur_range=(0.0, 0.2 * 255.0), noise_range=(0.0, 0.2 * 255.0)):
    # Apply random Gaussian blur and noise to each image in the batch
    def augment_image(image):
        # Apply Gaussian blur with random severity within `blur_range`
        blur_sigma = np.random.uniform(*blur_range)
        image = tfa.image.gaussian_filter2d(image, filter_shape=(3, 3), sigma=blur_sigma)

        # Add Gaussian noise with random severity within `noise_range`
        noise_scale = np.random.uniform(*noise_range)
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=noise_scale)
        image = tf.add(image, noise)

        return image

    augmented_batch = tf.map_fn(augment_image, batch)
    return augmented_batch


def prepare(ds, batch_size, shuffle=False, augment=False):
    if shuffle:
        ds = ds.shuffle(1000)

    # Batch all datasets.
    ds = ds.batch(batch_size)

    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x), y),
                    num_parallel_calls=AUTOTUNE)

    ds = ds.map(lambda x, y: (preprocess_input(x), preprocess_input(y)),
                num_parallel_calls=AUTOTUNE)
    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)


def load_and_preprocess_dataset(batch_size):
    data_dir = pathlib.Path(patch_dir)
    list_ds = tf.data.Dataset.list_files(str(data_dir / '*/*'), shuffle=False)
    image_count = len(list_ds)
    val_size = int(image_count * 0.2)
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)
    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    train_ds_aug = prepare(train_ds, batch_size, shuffle=True, augment=True)
    val_ds_aug = prepare(val_ds, batch_size)
    return train_ds_aug, val_ds_aug


class KimiaNetEncoder:
    """
    This class is used to create an encoder model based on the KimiaNet architecture.
    """

    def __init__(self):
        self.patch_dir = patch_dir
        self.network_weights_address = network_weights_address
        self.network_input_patch_width = network_input_patch_width
        self.img_format = img_format
        self.intermediate_features_names = ["pool2_pool", "pool3_pool", "pool4_pool"]
        self.intermediate_features = []
        self.model = self.create_encoder()

    def retrieve_intermediate_layers(self, model):
        ids = []
        for index, layers in enumerate(model.layers):
            if layers.name in self.intermediate_features_names:
                ids.append(index)
        return ids

    def create_encoder(self):
        dnx = DenseNet121(include_top=False, weights=self.network_weights_address,
                          input_shape=(self.network_input_patch_width, self.network_input_patch_width, 3))
        index = self.retrieve_intermediate_layers(dnx)
        output = [dnx.layers[i].output for i in index]
        self.intermediate_features = output
        encoder = Model(dnx.input, output, name='encoder')
        return encoder


class KimiaNetDecoder:
    """
    This class is used to create a decoder model based on the KimiaNet architecture.
    """

    def __init__(self):
        self.patch_dir = patch_dir
        self.network_weights_address = network_weights_address
        self.network_input_patch_width = network_input_patch_width
        self.img_format = img_format
        self.decoder_blocks = [24, 12, 6]
        self.transpose_feature_maps = [1024, 512, 256]
        self.transpose_block_number = 0
        self.growth_rate = 16
        self.model = self.create_decoder()

    def transition_block_transpose(self, x, name):
        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(x)
        x = layers.Activation('relu', name=name + '_relu')(x)
        x = layers.UpSampling2D(size=2, name=name + '_upsample')(x)
        x = layers.Conv2DTranspose(self.transpose_feature_maps[self.transpose_block_number], 1, use_bias=False,
                                   name=name + '_conv')(x)
        self.transpose_block_number += 1
        return x

    def dense_block_transpose(self, x, blocks, name):
        for i in range(blocks):
            x = conv_block(x, self.growth_rate, name=name + '_block' + str(i + 1))
        return x

    def create_decoder(self):
        encoder = KimiaNetEncoder()
        bn_axis = 3 if backend.image_data_format() == "channels_last" else 1
        i = encoder.intermediate_features[-1]
        x = self.transition_block_transpose(i, "transition_block1")
        x = self.dense_block_transpose(x, self.decoder_blocks[0], "dense_block1")
        x = layers.Concatenate(axis=-1)([x, encoder.intermediate_features[-2]])
        x = self.transition_block_transpose(x, "transition_block2")
        x = self.dense_block_transpose(x, self.decoder_blocks[1], "dense_block2")
        x = layers.Concatenate(axis=-1)([x, encoder.intermediate_features[-3]])
        x = self.transition_block_transpose(x, "transition_block3")
        x = self.dense_block_transpose(x, self.decoder_blocks[2], "dense_block3")
        x = layers.Conv2DTranspose(64, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
        x = layers.Activation("relu")(x)
        x = layers.UpSampling2D(size=4)(x)
        x = layers.Conv2DTranspose(3, 1, use_bias=False)(x)
        decoder = Model(encoder.intermediate_features, x, name="decoder")
        self.transpose_block_number = 0  # reset the counter
        return decoder


class KimiaNetAutoencoder:
    """
    This class is used to create an autoencoder model based on the KimiaNet architecture.
    """

    def __init__(self):
        self.patch_dir = patch_dir
        self.network_weights_address = network_weights_address
        self.network_input_patch_width = network_input_patch_width
        self.batch_size = None
        self.img_format = img_format
        self.model = self.create_autoencoder()

    def create_autoencoder(self):
        i = Input(shape=(self.network_input_patch_width, self.network_input_patch_width, 3), dtype=tf.float32)
        x = KimiaNetEncoder().model(i)
        x = KimiaNetDecoder().model(x)
        autoencoder = Model(i, x, name="autoencoder")
        return autoencoder

    def freeze_encoder(self):
        self.model.layers[-2].trainable = False

    def unfreeze_encoder(self):
        self.model.layers[-2].trainable = True

    def compile(self, lr=1e-4, loss_function='mse'):
        self.model.compile(optimizer=Adam(learning_rate=lr), loss=loss_function, metrics=['mse'])


def save_training_results(histories, index):
    histories[0]['loss'].extend(histories[1]['loss'])
    histories[0]['val_loss'].extend(histories[1]['val_loss'])
    history_dict = histories[0]
    with open(f'../results/loss_history_{index}.pickle', 'wb') as output_file:
        pickle.dump(history_dict, output_file, pickle.HIGHEST_PROTOCOL)


def train_autoencoder():
    date_time = datetime.now().strftime("%d%m%Y_%H:%M:%S")
    batch_size = 32
    lr = 1e-5
    histories = []
    train_dataset, val_dataset = load_and_preprocess_dataset(batch_size)
    autoencoder_ = KimiaNetAutoencoder()
    autoencoder_.freeze_encoder()
    autoencoder_.compile(lr=lr)
    callbacks_frozen = [
        tf.keras.callbacks.ModelCheckpoint(
            # Path where to save the model
            # The two parameters below mean that we will overwrite
            # the current checkpoint if and only if
            # the `val_loss` score has improved.
            # The saved model name will include the current epoch.
            filepath=f"../models/Model_{batch_size}_{lr}_{date_time}" + "/1.{epoch}",
            save_best_only=True,  # Only save a model if `val_loss` has improved.
            monitor="val_loss",
            verbose=1,
        ),
    ]
    history1 = autoencoder_.model.fit(train_dataset, epochs=20, validation_data=val_dataset,
                                      callbacks=callbacks_frozen, verbose=1)
    histories.append(history1.history)
    # Unfreeze the encoder
    autoencoder_.unfreeze_encoder()
    # Re-compile the model
    autoencoder_.compile(lr=lr)
    # Set up callbacks for saving the model and early stopping if the model doesn't improve
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"../models/Model_{batch_size}_{lr}_{date_time}" + "/2.{epoch}",
            save_best_only=True,  # Only save a model if `val_loss` has improved.
            monitor="val_loss",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.0001,
            patience=3,
            verbose=1,
        ),
    ]
    history2 = autoencoder_.model.fit(train_dataset, epochs=20, validation_data=val_dataset,
                                      callbacks=callbacks,
                                      verbose=1)
    histories.append(history2.history)
    save_training_results(histories, f"{batch_size}_{lr}_{date_time}")


if __name__ == "__main__":
    # train_autoencoder()
    encoder = KimiaNetEncoder().model.summary()
