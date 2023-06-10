import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=3072)],
        )
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

from tensorflow.python.client import device_lib
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from __utils__.mean_average_precision.mean_average_precision import (
    MeanAveragePrecision2d,
)
from __utils__ import spotting
from __utils__ import functions
import models

print("tf version:", tf.__version__)
print(tf.config.list_physical_devices("GPU"))
local_device_protos = device_lib.list_local_devices()
for x in local_device_protos:
    if x.device_type == "GPU":
        print(x.physical_device_desc)


def normalize_dev(image, label):
    image = tf.image.per_image_standardization(image)
    # label = tf.cast(label, tf.float32)
    return image, label


def generator(X, y, batch_size=12):
    while True:
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            num_images = end - start
            X[start:end] = functions.normalize(X[start:end])
            u = np.array(X[start:end])[:, :, :, 0].reshape(num_images, 42, 42, 1)
            v = np.array(X[start:end])[:, :, :, 1].reshape(num_images, 42, 42, 1)
            os = np.array(X[start:end])[:, :, :, 2].reshape(num_images, 42, 42, 1)
            # yield [u, v, os], np.array(y[start:end])

            # Fit for ViT
            u_v_os = np.concatenate((u, v, os), axis=3)
            yield u_v_os, np.array(y[start:end])


def generator_dev(X, y):
    start = 0
    while start < len(X):
        yield X[start], y[start]
        start += 1


def train(
    X,
    y,
    groups,
    expression_type,
    model_name,
    train_or_not,
    epochs,
    batch_size,
    clean_subjects_videos_ground_truth_labels,
    resampled_clean_videos_images_features,
    clean_subjects,
    clean_subjects_videos_code,
    k,
    show_plot_or_not,
):
    logo = LeaveOneGroupOut()
    n_splits = logo.get_n_splits(X, y, groups)
    split = 0
    reconstructed_clean_videos_ground_truth_labels_len = 0
    metric_fn = MeanAveragePrecision2d(num_classes=1)
    matrix = {"precision": [], "recall": [], "F1-score": []}
    p = 0.55  # From our analysis, 0.55 achieved the highest F1-Score

    # Leave One Subject Out
    for train_index, test_index in logo.split(X, y, groups):
        split += 1
        print(f"Split {split} / {n_splits} is in process.")

        # Get training set
        X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
        # Get testing set
        y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

        # To reset the model at every LOSO testing
        print("------Initializing Model-------")
        tf.keras.backend.clear_session()
        model = models.load_compile_model(model_name)

        if train_or_not is True:
            # downsampling
            X_train, y_train = functions.downsample(X_train, y_train)

            # Data augmentation to the micro-expression samples only
            if expression_type == "me":
                X_train, y_train = functions.augment_data(X_train, y_train)

            model.fit(
                generator(X_train, y_train, batch_size),
                steps_per_epoch=len(X_train) / batch_size,
                epochs=epochs,
                validation_data=generator(X_test, y_test, batch_size),
                validation_steps=len(X_test) / batch_size,
                shuffle=True,
            )

        else:
            # Load Pretrained Weights
            # model.load_weights(weights_path)
            pass

        pred = model.predict(
            generator(X_test, y_test, batch_size),
            steps=len(X_test) / batch_size,
        )

        # Spotting
        (reconstructed_clean_videos_ground_truth_labels_len, metric_fn) = spotting.spot(
            pred,
            reconstructed_clean_videos_ground_truth_labels_len,
            clean_subjects_videos_ground_truth_labels,
            split,
            resampled_clean_videos_images_features,
            clean_subjects,
            clean_subjects_videos_code,
            k,
            metric_fn,
            p,
            show_plot_or_not,
        )

        # Evaluation
        # every evaluation considers all splitted videos
        precision, recall, F1_score = functions.evaluate(
            reconstructed_clean_videos_ground_truth_labels_len,
            metric_fn,
        )

        matrix["precision"].append(precision)
        matrix["recall"].append(recall)
        matrix["F1-score"].append(F1_score)

        print(f"Split {split} / {n_splits} is processed.\n")

    return metric_fn, matrix
