import os
import cv2
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

from preprocessing import (
    create_dataset,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SEQUENCE_LENGTH,
    DATASET_DIR,
    seed_constant,
)


def create_convlstm_model(num_classes):
    """
    Constructs a ConvLSTM2D network for spatio-temporal video classification.

    num_classes must come from the actual dataset (len(classes_list) from
    create_dataset()), not a hardcoded list - that's the mismatch that broke
    training before.
    """
    model = Sequential()

    model.add(ConvLSTM2D(filters=4, kernel_size=(3, 3), activation='tanh',
                         data_format="channels_last",
                         recurrent_dropout=0.2, return_sequences=True,
                         input_shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))

    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(ConvLSTM2D(filters=8, kernel_size=(3, 3), activation='tanh',
                         data_format="channels_last",
                         recurrent_dropout=0.2, return_sequences=True))

    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(ConvLSTM2D(filters=14, kernel_size=(3, 3), activation='tanh',
                         data_format="channels_last",
                         recurrent_dropout=0.2, return_sequences=True))

    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='tanh',
                         data_format="channels_last",
                         recurrent_dropout=0.2, return_sequences=False))

    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()
    return model


if __name__ == "__main__":
    # Build features/labels straight from the BISINDO dataset_dir (flat
    # folder, WORD_NN.mp4 / labels.csv - see PROJECT_CONTEXT.md §6), with
    # classes_list discovered dynamically instead of assumed.
    features, labels, classes_list = create_dataset(DATASET_DIR)
    num_classes = len(classes_list)
    print(f"Classes found ({num_classes}): {classes_list}")

    one_hot_encoded_labels = to_categorical(labels, num_classes=num_classes)

    features_train, features_test, labels_train, labels_test = train_test_split(
        features,
        one_hot_encoded_labels,
        test_size=0.25,
        shuffle=True,
        random_state=seed_constant,
    )

    convlstm_model = create_convlstm_model(num_classes=num_classes)

    early_stopping_callback = EarlyStopping(
        monitor='val_loss', patience=10, mode='min', restore_best_weights=True
    )

    convlstm_model.compile(
        loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"]
    )

    convlstm_model_training_history = convlstm_model.fit(
        x=features_train,
        y=labels_train,
        epochs=50,
        batch_size=4,
        shuffle=True,
        validation_split=0.2,
        callbacks=[early_stopping_callback],
    )

    model_evaluation_history = convlstm_model.evaluate(features_test, labels_test)