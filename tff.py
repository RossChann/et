import tensorflow as tf
import numpy as np
from utils import port_datasets
from utils import port_pretrained_models
from selection_solver_DP import selection_DP, downscale_t_dy_and_t_dw
from profiler import profile_parser
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import tensorflow_federated as tff
import time
from tqdm import tqdm
import os
from utils import clear_cache_and_rec_usage




def create_tff_model(model_type, input_shape, num_classes):
    def model_fn():
        # 调用你的模型导入函数获取 Keras 模型
        keras_model = port_pretrained_models(model_type, input_shape, num_classes)
        return tff.learning.from_keras_model(
            keras_model,
            input_spec=input_shape,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )

    # 创建 TFF 模型
    tff_model = tff.learning.framework.build_model(model_fn)
    return tff_model

def port_datasets(dataset_name, input_shape=28, batch_size=4, num_split=2):
    """
    This function loads the train and test splits of the requested dataset, and
    creates input pipelines for training in TFF format.

    Args:
        dataset_name (str): name of the dataset
        input_shape (tuple): NN input shape excluding batch dim
        batch_size (int): batch size of training split,
        default batch size for testing split is batch_size*2
        num_split (int): number of splits for the training dataset

    Raises:
        NotImplementedError: The requested dataset is not implemented

    Returns:
        Train and test splits of the request dataset in TFF format
    """

    # maximize number limit of opened files
    low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

    def prep(x, y):
        x = tf.image.resize(x, [input_shape[0], input_shape[1]])
        return x, y

    def client_data(client_id):
        return client_datasets[client_id]

    if dataset_name in ['caltech_birds2011', 'stanford_dogs', 'oxford_iiit_pet']:
        splits = tfds.even_splits('train', n=num_split)
        client_datasets = []

        for split in splits:
            ds_train = tfds.load(dataset_name, split=split, as_supervised=True)
            ds_train = ds_train.map(prep, num_parallel_calls=tf.data.AUTOTUNE) \
                .batch(batch_size) \
                .prefetch(buffer_size=tf.data.AUTOTUNE)
            client_datasets.append(ds_train)

        ds_test = tfds.load(dataset_name, split='test', as_supervised=True)

        ds_test = ds_test.map(prep, num_parallel_calls=tf.data.AUTOTUNE) \
            .batch(batch_size * 2) \
            .prefetch(buffer_size=tf.data.AUTOTUNE)

    elif dataset_name == 'mnist-noniid':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # 数据预处理
        x_train = np.repeat(x_train[..., np.newaxis], 3, axis=-1)  # 复制通道以将灰度图像转换为RGB图像
        x_test = np.repeat(x_test[..., np.newaxis], 3, axis=-1)
        x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant')  # 填充图像以达到32x32的大小
        x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant')
        x_train = x_train.astype('float32') / 255.0  # 归一化像素值
        x_test = x_test.astype('float32') / 255.0

        # 根据标签将训练数据集划分为4个子集
        client_datasets = [[] for _ in range(num_split)]
        for x, y in zip(x_train, y_train):
            client_id = y // 5
            client_datasets[client_id].append((x, y))

        # 将每个客户端的数据转换为tf.data.Dataset
        for i in range(num_split):
            client_images = [x for x, _ in client_datasets[i]]
            client_labels = [y for _, y in client_datasets[i]]
            client_images = np.array(client_images)
            client_labels = np.array(client_labels)
            client_datasets[i] = tf.data.Dataset.from_tensor_slices((client_images, client_labels))
            client_datasets[i] = client_datasets[i].map(prep, num_parallel_calls=tf.data.AUTOTUNE) \
                .batch(batch_size) \
                .prefetch(buffer_size=tf.data.AUTOTUNE)

        # 创建测试数据集
        ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        ds_test = ds_test.map(prep, num_parallel_calls=tf.data.AUTOTUNE) \
            .batch(batch_size * 2) \
            .prefetch(buffer_size=tf.data.AUTOTUNE)

    elif dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # 数据预处理
        x_train = np.repeat(x_train[..., np.newaxis], 3, axis=-1)  # 复制通道以将灰度图像转换为RGB图像
        x_test = np.repeat(x_test[..., np.newaxis], 3, axis=-1)
        x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant')  # 填充图像以达到32x32的大小
        x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant')
        x_train = x_train.astype('float32') / 255.0  # 归一化像素值
        x_test = x_test.astype('float32') / 255.0

        # 将训练数据集划分为多个子集
        client_datasets = []
        split_indices = np.array_split(np.arange(len(x_train)), num_split)

        for indices in split_indices:
            client_images = x_train[indices]
            client_labels = y_train[indices]
            ds_client = tf.data.Dataset.from_tensor_slices((client_images, client_labels))
            ds_client = ds_client.map(prep, num_parallel_calls=tf.data.AUTOTUNE) \
                .batch(batch_size) \
                .prefetch(buffer_size=tf.data.AUTOTUNE)
            client_datasets.append(ds_client)

        # 创建测试数据集
        ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        ds_test = ds_test.map(prep, num_parallel_calls=tf.data.AUTOTUNE) \
            .batch(batch_size * 2) \
            .prefetch(buffer_size=tf.data.AUTOTUNE)

    elif dataset_name == 'cifar10-noniid':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        # 数据预处理
        x_train = x_train.astype('float32') / 255.0  # 归一化像素值
        x_test = x_test.astype('float32') / 255.0

        # 根据标签将训练数据集划分为4个子集
        client_datasets = [[] for _ in range(num_split)]
        for x, y in zip(x_train, y_train):
            client_id = y[0] // (10 // num_split)  # CIFAR-10有10个类别,平均分成4组
            client_datasets[client_id].append((x, y[0]))

        # 将每个客户端的数据转换为tf.data.Dataset
        for i in range(num_split):
            client_images = [x for x, _ in client_datasets[i]]
            client_labels = [y for _, y in client_datasets[i]]
            client_images = np.array(client_images)
            client_labels = np.array(client_labels)
            client_datasets[i] = tf.data.Dataset.from_tensor_slices((client_images, client_labels))
            client_datasets[i] = client_datasets[i].map(prep, num_parallel_calls=tf.data.AUTOTUNE) \
                .batch(batch_size) \
                .prefetch(buffer_size=tf.data.AUTOTUNE)

        # 创建测试数据集
        ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        ds_test = ds_test.map(prep, num_parallel_calls=tf.data.AUTOTUNE) \
            .batch(batch_size * 2) \
            .prefetch(buffer_size=tf.data.AUTOTUNE)

    elif dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        # 数据预处理
        x_train = x_train.astype('float32') / 255.0  # 归一化像素值
        x_test = x_test.astype('float32') / 255.0

        # 将训练数据集划分为多个子集
        client_datasets = []
        split_indices = np.array_split(np.arange(len(x_train)), num_split)

        for indices in split_indices:
            client_images = x_train[indices]
            client_labels = y_train[indices]
            ds_client = tf.data.Dataset.from_tensor_slices((client_images, client_labels))
            ds_client = ds_client.map(prep, num_parallel_calls=tf.data.AUTOTUNE) \
                .batch(batch_size) \
                .prefetch(buffer_size=tf.data.AUTOTUNE)
            client_datasets.append(ds_client)

        # 创建测试数据集
        ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        ds_test = ds_test.map(prep, num_parallel_calls=tf.data.AUTOTUNE) \
            .batch(batch_size * 2) \
            .prefetch(buffer_size=tf.data.AUTOTUNE)

    else:
        raise NotImplementedError("This dataset has not been implemented yet")

    federated_train_data = tff.simulation.ClientData.from_clients_and_fn(
        client_ids=list(range(num_split)),
        create_tf_dataset_for_client_fn=client_data
    )

    return federated_train_data, ds_test

@tff.tf_computation
def client_update(model, dataset, run_name, logdir, timing_info, optim, lr, weight_decay, epochs, interval, rho, disable_random_id, save_model, save_txt):
    def elastic_training(model, ds_train, ds_test):
        # 这里是你原始的elastic_training函数的内容,保持不变
        ...

    elastic_training(model, dataset['train'], dataset['test'])
    return model.trainable_weights



model_type = 'resnet50'
input_shape = (224, 224, 3)
num_classes = 37