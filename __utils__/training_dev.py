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
            [tf.config.LogicalDeviceConfiguration(memory_limit=4608)],
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


def train(
    dataset_dir,
    clean_subjects,
    y,
    expression_type,
    model_name,
    train_or_not,
    epochs,
    batch_size,
):
    preds = []

    # Leave One Subject Out
    for split, split_clean_subject in enumerate(clean_subjects):
        print(f"Split {split} / {len(clean_subjects)} is in process.")
        # To reset the model at every LOSO testing
        print("------Initializing Model-------")
        tf.keras.backend.clear_session()
        model = models.load_compile_model(model_name)

        # get test dataset
        X_test = functions.load_subject_pkl_file(
            expression_type, dataset_dir, subject=split_clean_subject
        )
        y_test = y[split]

        # test normalization
        # cv2.normalize works better than tf.image
        X_test = functions.normalize(X_test)

        val_ds = (
            tf.data.Dataset.from_tensor_slices((X_test, y_test))
            # .map(
            #     normalize_dev,
            #     num_parallel_calls=tf.data.AUTOTUNE,
            # )
            .batch(batch_size).prefetch(tf.data.AUTOTUNE)
        )

        for clean_subject_index, clean_subject in enumerate(clean_subjects):
            if clean_subject_index != split:
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
                        validation_data=val_ds,
                        shuffle=True,
                        # callbacks=[callback],
                    )
                    del X_train
                    gc.collect()
                else:
                    # Load Pretrained Weights
                    # model.load_weights(weights_path)
                    pass

        pred = model.predict(val_ds)
        preds.append(pred)
        del X_test
        gc.collect()
        print(f"Split {split} / {len(clean_subjects)} is processed.\n")

    return preds
