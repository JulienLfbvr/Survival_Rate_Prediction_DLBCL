from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from keras.applications import DenseNet121
from keras.applications.densenet import conv_block
from keras import Input
from tensorflow.keras.applications.densenet import preprocess_input
from keras import backend
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import pandas as pd
import os
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import PIL
import PIL.Image
import pathlib
import tensorflow_addons as tfa

patch_dir = "D:\\ISEN\\M1\\Projet M1\\DLBCL-Morph\\Patches\\HE"
extracted_features_save_adr = "../extracted_features.pickle"
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


# def create_train_dataset(patch_dir_):
#     patient_df = pd.read_csv('D:\\ISEN\\M1\\Projet M1\\KimiaNet\\CSV\\clinical_data_with_no_missing_values.csv')
#     y_data = patient_df[['patient_id', 'Follow-up Status']].copy()
#     train_dataset = []
#     for dirs in os.listdir(patch_dir_):
#         patient_id = dirs
#         for files in os.listdir(patch_dir_ + "\\" + dirs):
#             if files.endswith(".png"):
#                 image_data = skimage.io.imread(f_name=patch_dir_ + "\\" + dirs + "\\" + files)
#                 train_dataset.append([image_data, patient_id])
#
#     x_train = []
#     y_train = []
#     for i in range(len(train_dataset)):
#         x_train.append(train_dataset[i][0])
#         y_data.loc[:, 'patient_id'] = y_data['patient_id'].astype(str)
#         y_train.append(y_data[y_data['patient_id'] == train_dataset[i][1]]['Follow-up Status'].values[0])
#     return x_train, y_train

# def create_train_dataset(patch_dir_):
#     train_dataset = []
#     for dirs in os.listdir(patch_dir_):
#         for files in os.listdir(patch_dir_ + "\\" + dirs):
#             if files.endswith(".png"):
#                 image_data = skimage.io.imread(f_name=patch_dir_ + "\\" + dirs + "\\" + files)
#                 train_dataset.append(image_data)
#     return train_dataset


class KimiaNetEncoder:
    """
    This class is used to create an encoder model based on the KimiaNet architecture.
    """

    def __init__(self):
        self.patch_dir = patch_dir
        self.extracted_features_save_adr = extracted_features_save_adr
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


# def conv_block_transpose(x, growth_rate, name):
#     bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
#     x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(x)
#     x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
#     x1 = layers.Conv2D(4 * growth_rate, 1, use_bias=False, name=name + '_1_conv')(x1)
#     x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x1)
#     x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
#     x1 = layers.Conv2D(growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(x1)
#     x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
#     return x


class KimiaNetDecoder:
    """
    This class is used to create a decoder model based on the KimiaNet architecture.
    """

    def __init__(self):
        self.patch_dir = patch_dir
        self.extracted_features_save_adr = extracted_features_save_adr
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
        self.extracted_features_save_adr = extracted_features_save_adr
        self.network_weights_address = network_weights_address
        self.network_input_patch_width = network_input_patch_width
        self.batch_size = None
        self.visited_patches_paths = []
        self.img_format = img_format
        self.encoder = None
        self.decoder = None
        self.model = self.create_autoencoder()

    def create_autoencoder(self):
        i = Input(shape=(network_input_patch_width, network_input_patch_width, 3), dtype=tf.float32)
        # x = tf.cast(i, 'float32')
        # x = preprocess_input(x)
        # # add noise
        x = KimiaNetEncoder().model(i)
        self.encoder = x
        x = KimiaNetDecoder().model(x)
        self.decoder = x
        autoencoder = Model(i, x, name="autoencoder")
        return autoencoder

    def freeze_encoder(self):
        self.model.layers[-2].trainable = False

    def unfreeze_encoder(self):
        self.model.layers[-2].trainable = True

    def compile(self, lr=1e-4, loss_function='mse'):
        self.model.compile(optimizer=Adam(learning_rate=lr), loss=loss_function, metrics=['mse'])

    # def create_batch(self):
    #     batch = []
    #     patch_nb = 0
    #     patch_dir_ = self.patch_dir
    #     for dirs in os.listdir(patch_dir_):
    #         if patch_nb == self.batch_size:
    #             break
    #         for files in os.listdir(patch_dir_ + "\\" + dirs):
    #             if files.endswith(".png") and files not in self.visited_patches_paths:
    #                 image_data = skimage.io.imread(f_name=patch_dir_ + "\\" + dirs + "\\" + files)
    #                 batch.append(image_data)
    #                 self.visited_patches_paths.append(files)
    #                 patch_nb += 1
    #                 if patch_nb == self.batch_size:
    #                     break
    #     return batch

    # def train(self, epochs, batch_size):
    #     self.batch_size = batch_size
    #     self.compile()
    #
    #     # Freeze the encoder
    #     self.freeze_encoder()
    #
    #     # Load the dataset
    #     X = create_train_dataset(self.patch_dir)
    #     x_train, x_test = train_test_split(X, test_size=0.2, random_state=42)
    #
    #     # Divide the dataset into batches
    #     x_train = np.array_split(x_train, len(x_train) / self.batch_size)
    #     x_test = np.array_split(x_test, len(x_test) / self.batch_size)
    #
    #     # Train the model
    #     for i in range(epochs):
    #         print("Epoch: ", i)
    #         for j in range(len(x_train)):
    #             self.model.train_on_batch(
    #                 x_train[j],
    #                 x_train[j],
    #                 sample_weight=None,
    #                 class_weight=None,
    #                 reset_metrics=True,
    #                 return_dict=False,
    #             )
    #
    #         for j in range(len(x_test)):
    #             self.model.test_on_batch(
    #                 x_test[j],
    #                 x_test[j],
    #                 sample_weight=None,
    #                 reset_metrics=True,
    #                 return_dict=False,
    #             )


# KimiaNetEncoder().model.summary()
# KimiaNetDecoder().model.summary()
# KimiaNetAutoencoder().model.layers[-2].summary()
# KimiaNetAutoencoder().model.layers[-1].summary()


def save_training_results(history, index, frozen):
    # Plot training & validation loss values
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # Create the directory if it does not exist
    if not os.path.exists(f"../results/model_{index}"):
        os.makedirs(f"../results/model_{index}")
    if frozen:
        plt.savefig(f"../results/model_{index}/loss_frozen.png")
    else:
        plt.savefig(f"../results/model_{index}/loss.png")
    plt.close()


def train_autoencoder():
    batch_sizes = [16, 32, 64]
    learning_rates = [1e-4, 1e-5, 1e-6]
    for batch_size in batch_sizes:
        for lr in learning_rates:
            train_dataset, val_dataset = load_and_preprocess_dataset(batch_size)
            autoencoder_ = KimiaNetAutoencoder()
            autoencoder_.freeze_encoder()
            autoencoder_.compile(lr=lr)
            callbacks_frozen = [
                keras.callbacks.ModelCheckpoint(
                    # Path where to save the model
                    # The two parameters below mean that we will overwrite
                    # the current checkpoint if and only if
                    # the `val_loss` score has improved.
                    # The saved model name will include the current epoch.
                    filepath=f"../models/Model_{batch_size}_{lr}" + "/1.{epoch}",
                    save_best_only=True,  # Only save a model if `val_loss` has improved.
                    monitor="val_loss",
                    verbose=1,
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta=0.005,
                    patience=1,
                    verbose=1,
                ),
            ]
            history = autoencoder_.model.fit(train_dataset, epochs=10, validation_data=val_dataset,
                                             callbacks=callbacks_frozen, verbose=1)
            save_training_results(history, f"{batch_size}_{lr}", True)
            # Unfreeze the encoder
            autoencoder_.unfreeze_encoder()
            # Re-compile the model
            autoencoder_.compile(lr=lr)
            # Set up callbacks for saving the model and early stopping if the model doesn't improve
            callbacks = [
                keras.callbacks.ModelCheckpoint(
                    filepath=f"../models/Model_{batch_size}_{lr}" + "/2.{epoch}",
                    save_best_only=True,  # Only save a model if `val_loss` has improved.
                    monitor="val_loss",
                    verbose=1,
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta=0.002,
                    patience=2,
                    verbose=1,
                ),
            ]
            history = autoencoder_.model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=callbacks,
                                             verbose=1)
            save_training_results(history, f"{batch_size}_{lr}", False)


if __name__ == "__main__":
    train_autoencoder()

    # train_dataset, val_dataset = load_and_preprocess_dataset(16)
    # autoencoder_ = KimiaNetAutoencoder()
    # autoencoder_.freeze_encoder()
    # autoencoder_.compile()
    # callbacks = [
    #     keras.callbacks.ModelCheckpoint(
    #         # Path where to save the model
    #         # The two parameters below mean that we will overwrite
    #         # the current checkpoint if and only if
    #         # the `val_loss` score has improved.
    #         # The saved model name will include the current epoch.
    #         filepath="../models/myModel_1.{epoch}",
    #         save_best_only=True,  # Only save a model if `val_loss` has improved.
    #         monitor="val_loss",
    #         verbose=1,
    #     )
    # ]
    # history = autoencoder_.model.fit(train_dataset, epochs=1, validation_data=val_dataset, callbacks=callbacks,
    #                                  verbose=1)
    # # Plot training & validation loss values
    # save_training_results(history, 1)
    # # Unfreeze the encoder
    # autoencoder_.unfreeze_encoder()
    # # Re-compile the model
    # autoencoder_.compile()
    # # Set up callbacks for saving the model and early stopping if the model doesn't improve
    # callbacks = [
    #     keras.callbacks.ModelCheckpoint(
    #         filepath="../models/myModel_2.{epoch}",
    #         save_best_only=True,  # Only save a model if `val_loss` has improved.
    #         monitor="val_loss",
    #         verbose=1,
    #     ),
    #     keras.callbacks.EarlyStopping(
    #         monitor="val_loss",
    #         min_delta=0.005,
    #         patience=2,
    #         verbose=1,
    #     ),
    # ]
    # history = autoencoder_.model.fit(train_dataset, epochs=20, validation_data=val_dataset, callbacks=callbacks,
    #                                  verbose=1)
    # # Plot training & validation loss values
    # save_training_results(history, 1)
