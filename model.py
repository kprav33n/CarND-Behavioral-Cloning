#!/usr/bin/env python

from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import os
import pandas


class DataSource(object):
    def __init__(self):
        data_path = 'data'
        driving_log_path = os.path.join(data_path, 'driving_log.csv')
        driving_log = pandas.read_csv(driving_log_path)

        # Extract center, right, and left camera images.
        center_image_paths = list(
            map(lambda s: os.path.join(data_path, s),
                driving_log['center'].str.strip().tolist()))
        right_image_paths = list(
            map(lambda s: os.path.join(data_path, s),
                driving_log['right'].str.strip().tolist()))
        left_image_paths = list(
            map(lambda s: os.path.join(data_path, s),
                driving_log['left'].str.strip().tolist()))

        # Extract steering angle, and interpolate right and left camera
        # steering angles.
        center_steering_angles = driving_log['steering'].tolist()
        right_steering_angles = list(map(lambda x: x + 0.25,
                                         center_steering_angles))
        left_steering_angles = list(map(lambda x: x + 0.25,
                                        center_steering_angles))

        # Combine all data.
        image_paths = center_image_paths + right_image_paths + left_image_paths
        for path in image_paths:
            if not os.path.exists(path):
                raise UserWarning(path)
        steering_angles = (center_steering_angles +
                           right_steering_angles +
                           left_steering_angles)
        image_paths, steering_angles = shuffle(image_paths, steering_angles,
                                               random_state=0)
        (self.train_image_paths,
         self.validation_image_paths,
         self.train_steering_angles,
         self.validation_steering_angles) = train_test_split(
             image_paths, steering_angles, test_size=0.2, random_state=0)
        self.num_train_samples = len(self.train_image_paths)
        self.num_validation_samples = len(self.validation_image_paths)

    def train_generator(self, batch_size):
        offset = 0
        while True:
            if offset == 0:
                self.train_image_paths, self.train_steering_angles = shuffle(
                    self.train_image_paths, self.train_steering_angles)
            if offset + batch_size >= self.num_train_samples:
                end_offset = -1
            else:
                end_offset = offset + batch_size
            images = np.asarray(
                [cv2.imread(path)
                 for path in self.train_image_paths[offset:end_offset]]
            )
            angles = np.asarray(self.train_steering_angles[offset:end_offset])
            if end_offset == -1:
                offset = 0
            else:
                offset = end_offset

            yield (images, angles)

    def validation_generator(self, batch_size):
        offset = 0
        while True:
            if offset == 0:
                (self.validation_image_paths,
                 self.validation_steering_angles) = shuffle(
                    self.validation_image_paths,
                     self.validation_steering_angles)
            if offset + batch_size >= self.num_validation_samples:
                end_offset = -1
            else:
                end_offset = offset + batch_size
            images = np.asarray(
                [cv2.imread(path)
                 for path in self.validation_image_paths[offset:end_offset]]
            )
            angles = np.asarray(
                self.validation_steering_angles[offset:end_offset])
            if end_offset == -1:
                offset = 0
            else:
                offset = end_offset

            yield (images, angles)


def build_model():
    input_shape = (160, 320, 3)
    keep_prob = 0.5
    model = Sequential()

    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Convolution2D(3, 1, 1, border_mode='same'))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='elu'))
    model.add(Dropout(keep_prob))

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='elu'))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='elu'))
    model.add(Dropout(keep_prob))

    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), activation='elu'))

    model.add(Convolution2D(96, 3, 3, activation='elu'))
    model.add(Dropout(keep_prob))

    model.add(Convolution2D(96, 3, 3, activation='elu'))

    model.add(Convolution2D(128, 3, 3, activation='elu'))
    model.add(Dropout(keep_prob))

    model.add(Flatten())
    model.add(Dense(768, activation='elu'))
    model.add(Dense(256, activation='elu'))
    model.add(Dropout(keep_prob))
    model.add(Dense(128, activation='elu'))
    model.add(Dense(64, activation='elu'))
    model.add(Dropout(keep_prob))
    model.add(Dense(32, activation='elu'))
    # model.add(Dense(1, activation='tanh'))
    model.add(Dense(1))
    return model


def train_model(model):
    model.compile(loss='mean_squared_error', optimizer='adam')
    batch_size = 128
    nb_epoch = 2
    data_source = DataSource()
    model.fit_generator(
        data_source.train_generator(batch_size), data_source.num_train_samples,
        nb_epoch,
        validation_data=data_source.validation_generator(batch_size),
        nb_val_samples=data_source.num_validation_samples,
        verbose=1)


def save_model(model):
    model_file_path = 'model.json'
    weights_file_path = 'model.h5'
    model_json = model.to_json()
    with open(model_file_path, 'w') as model_file:
        model_file.write(model_json)
    model.save_weights(weights_file_path)


def main():
    model = build_model()
    print(model.summary())
    train_model(model)
    save_model(model)


if __name__ == '__main__':
    main()
