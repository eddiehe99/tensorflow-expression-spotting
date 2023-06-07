import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
from sklearn.model_selection import LeaveOneGroupOut
from mean_average_precision.mean_average_precision import MeanAveragePrecision2d
from spotting import *
from evaluation import *
import functions
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
    logo.get_n_splits(X, y, groups)
    split = 0
    reconstructed_clean_videos_ground_truth_labels_len = 0
    metric_fn = MeanAveragePrecision2d(num_classes=1)
    matrix = {"precision": [], "recall": [], "F1-score": []}
    p = 0.55  # From our analysis, 0.55 achieved the highest F1-Score

    # Leave One Subject Out
    for train_index, test_index in logo.split(X, y, groups):
        split += 1
        print(f"Split {split} is in process.")

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
            if expression_type == "micro-expression":
                X_train, y_train = functions.augment_data(X_train, y_train)

            X_train = functions.normalize(X_train)
            X_test = functions.normalize(X_test)

            train_ds = (
                tf.data.Dataset.from_tensor_slices((X_train, y_train))
                # .map(
                #     normalize_dev,
                #     num_parallel_calls=tf.data.AUTOTUNE,
                # )
                .batch(batch_size)
                .shuffle(len(X_train))
                .prefetch(tf.data.AUTOTUNE)
            )
            val_ds = (
                tf.data.Dataset.from_tensor_slices((X_test, y_test))
                # .map(
                #     normalize_dev,
                #     num_parallel_calls=tf.data.AUTOTUNE,
                # )
                .batch(batch_size).prefetch(tf.data.AUTOTUNE)
            )

            def scheduler(epoch, lr):
                lr_new = lr * (0.97**epoch)
                return lr_new if lr_new >= 5e-5 else 5e-5

            # callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
            callback = tf.keras.callbacks.LearningRateScheduler(
                schedule=lambda epoch: 0.001 * (0.9**epoch)
            )

            # with tf.device('device:GPU:1'):
            model.fit(
                train_ds,
                epochs=epochs,
                validation_data=val_ds,
                shuffle=True,
                # callbacks=[callback],
            )
        else:
            # Load Pretrained Weights
            # model.load_weights(weights_path)
            pass

        pred = model.predict(val_ds)

        # Spotting
        (reconstructed_clean_videos_ground_truth_labels_len, metric_fn) = spot(
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
        precision, recall, F1_score = evaluate(
            reconstructed_clean_videos_ground_truth_labels_len,
            metric_fn,
        )

        matrix["precision"].append(precision)
        matrix["recall"].append(recall)
        matrix["F1-score"].append(F1_score)

        print(f"Split {split} is processed.\n")

    return metric_fn, matrix
