import tensorflow_datasets as tfds
import tensorflow as tf

def get_cifar10_datasets(batch_size: int = 64):
    # Prevent TF from acquiring all GPU memory during training
    tf.config.set_visible_devices([], 'GPU')
    
    def preprocess(x):
        image = tf.cast(x['image'], tf.float32) / 255.0
        # Normalize with standard CIFAR-10 statistics
        mean = tf.constant([0.4914, 0.4822, 0.4465])
        std = tf.constant([0.2023, 0.1994, 0.2010])
        image = (image - mean) / std
        return {'image': image, 'label': x['label']}

    train_ds = tfds.load('cifar10', split='train', as_supervised=False)
    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(10000).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    test_ds = tfds.load('cifar10', split='test', as_supervised=False)
    test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds