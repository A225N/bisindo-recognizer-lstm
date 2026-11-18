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

# Pull the dataset pipeline + constants straight from preprocessing.py so this
# file can never drift out of sync with it (this is the same bug that broke
# convlstm.py: a hardcoded CLASSES_LIST with 4 UCF50 classes vs. the real
# BISINDO dataset having 12 classes -> Dense output size mismatch).
from preprocessing import (
    create_dataset,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SEQUENCE_LENGTH,
    DATASET_DIR,
    seed_constant,
)


from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

def _vgg16_preprocess_from_unit_range(x):
    """
    preprocessing.py normalizes frames to [0, 1] (frame / 255.0), but VGG16
    with ImageNet weights expects raw BGR pixel values with the ImageNet
    per-channel mean subtracted (tf.keras.applications.vgg16.preprocess_input).
    This undoes the /255 scaling and applies that expected preprocessing so
    the frozen VGG16 backbone receives inputs in the format it was trained
    on - same intent as VGG16FeatureExtractor in PROJECT_CONTEXT.md's main
    pipeline, just applied here to a whole frame instead of a MediaPipe hand
    crop.
    """
    return preprocess_input(x * 255.0)
 
 
def create_LRCN_model(num_classes):
    """
    LRCN model using a frozen VGG16 (ImageNet weights, no top) as the
    per-frame CNN feature extractor - mirroring PROJECT_CONTEXT.md's
    VGG16FeatureExtractor (frozen conv trunk + global average pool ->
    512-dim per-frame feature). Applied frame-by-frame across the sequence
    via TimeDistributed, then fed into an LSTM.
 
    Difference from PROJECT_CONTEXT.md's main pipeline: this runs on the
    WHOLE frame produced by preprocessing.py, since this script has no
    MediaPipe hand-detection/cropping step - PROJECT_CONTEXT.md's version
    runs VGG16 on a cropped hand region instead.
 
    num_classes must come from the actual dataset (len(classes_list) from
    create_dataset()), not a hardcoded list.
    """
    vgg16_base = VGG16(weights='imagenet', include_top=False,
                        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    vgg16_base.trainable = False  # frozen, same choice as the main pipeline
 
    model = Sequential()
 
    # Undo preprocessing.py's /255 scaling and re-apply VGG16's own
    # expected preprocessing, per-frame.
    model.add(TimeDistributed(
        Lambda(_vgg16_preprocess_from_unit_range),
        input_shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    ))
 
    # Frozen VGG16 applied independently to every frame in the sequence
    model.add(TimeDistributed(vgg16_base))
    model.add(TimeDistributed(GlobalAveragePooling2D()))
 
    # Recurrent LSTM Layer over the sequence of 512-dim VGG16 features
    model.add(LSTM(32))
 
    # Dense Output Layer
    model.add(Dense(num_classes, activation='softmax'))
 
    model.summary()
    return model
 
 
if __name__ == "__main__":
    # Build features/labels straight from the BISINDO dataset_dir (flat
    # folder, WORD_NN.mp4 / labels.csv - see PROJECT_CONTEXT.md §6), with
    # classes_list discovered dynamically instead of assumed.
    num_classes = len(classes_list)
    print(f"Classes found ({num_classes}): {classes_list}")
 
    one_hot_encoded_labels = to_categorical(labels, num_classes=num_classes)
 
    early_stopping_callback = EarlyStopping(
        monitor='val_loss', patience=10, mode='min', restore_best_weights=True
    )
 
    lrcn_model = create_LRCN_model(num_classes=num_classes)
 
    lrcn_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])
 
    lrcn_model_training_history = lrcn_model.fit(
        x=features_train, y=labels_train,
        epochs=70, batch_size=4,
        shuffle=True, validation_split=0.2,
        callbacks=[early_stopping_callback]
    )
 
    # Evaluate LRCN Model
    model_evaluation_history = lrcn_model.evaluate(features_test, labels_test)