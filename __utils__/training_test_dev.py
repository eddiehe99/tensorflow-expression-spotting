import tensorflow as tf
from tensorflow.python.client import device_lib
import gc
from __utils__ import functions
import models

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    # Restrict TensorFlow to only allocate X GB of memory on the first GPU
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


def train_and_test(
    dataset_dir,
    test_dataset_dir,
    clean_subjects,
    test_videos_name,
    y,
    expression_type,
    model_name,
    train_or_not,
    epochs,
    batch_size,
):
    preds = []
    print("------Initializing Model-------")
    tf.keras.backend.clear_session()
    model = models.load_compile_model(model_name)

    print("------Training-------")
    for clean_subject_index, clean_subject in enumerate(clean_subjects):
        print(
            f"{clean_subject} {clean_subject_index} / {len(clean_subjects)} is in training."
        )

        # Get training set
        X_train = functions.load_subject_pkl_file(
            expression_type, dataset_dir, subject=clean_subject
        )
        y_train = y[clean_subject_index]

        if train_or_not is True:
            # downsampling
            X_train, y_train = functions.downsample(X_train, y_train)

            # Data augmentation to the micro-expression samples only
            if expression_type == "me":
                X_train, y_train = functions.augment_data(X_train, y_train)

            # training normalization
            # cv2.normalize works better than tf.image
            X_train = functions.normalize(X_train)

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

            def scheduler(epoch, lr):
                lr_new = lr * (0.97**epoch)
                return lr_new if lr_new >= 5e-5 else 5e-5

            # callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
            callback = tf.keras.callbacks.LearningRateScheduler(
                schedule=lambda epoch: 0.001 * (0.9**epoch)
            )

            model.fit(
                train_ds,
                epochs=epochs,
                shuffle=True,
                # callbacks=[callback],
            )
            del X_train
            gc.collect()
        else:
            # Load Pretrained Weights
            # model.load_weights(weights_path)
            pass

        print(
            f"{clean_subject} {clean_subject_index} / {len(clean_subjects)} finishes training.\n"
        )

    print("------Test-------")
    for test_video_name in test_videos_name:
        print(f"Testing {test_video_name}")
        # Get test set
        X_test = functions.load_test_video_pkl_file(
            expression_type, test_dataset_dir, test_video_name
        )

        # normalization
        # cv2.normalize works better than tf.image
        X_test = functions.normalize(X_test)

        val_ds = (
            tf.data.Dataset.from_tensor_slices(X_test)
            # .map(
            #     normalize_dev,
            #     num_parallel_calls=tf.data.AUTOTUNE,
            # )
            .batch(batch_size).prefetch(tf.data.AUTOTUNE)
        )
        pred = model.predict(val_ds)
        preds.append(pred)
        del X_test
        gc.collect()
        print(f"Finish testing {test_video_name}")

    print("All training and test are done.")
    return preds
