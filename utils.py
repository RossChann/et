import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
from vit_keras import vit
from vit_utils import vit_b16
import resource
import gc
from subprocess import Popen, PIPE
from threading import Timer
import sys
import os
import numpy as np

def my_bool(s):
    return s != 'False'

class RepeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)

def _clear_mem_cache():
    # os.system('/bin/bash -c "sync ; echo 1 > /proc/sys/vm/drop_caches ; "') 
    # os.system('/bin/bash -c "sync ; echo 2 > /proc/sys/vm/drop_caches ; "') 
    # os.system('/bin/bash -c "sync ; echo 3 > /proc/sys/vm/drop_caches ; "') 
    return

def _print_mem_free():
    process_free = Popen(["free"], stdout=PIPE)
    (output, err) = process_free.communicate()
    exit_code = process_free.wait()
    output_string = output.decode('UTF-8')
#    output_file.write(str(time.time()))
#    output_file.write(output_string)
#    output_file.write('\n')
#    output_file.flush()

def clear_cache_and_rec_usage():
    # NOOP
    return

def record_once():
    # _clear_mem_cache()
    gc.collect()
    # _print_mem_free()


# timer = RepeatTimer(15, record_once)
# timer.start()

def sig_stop_handler(sig, frame):
    global timer
    # timer.cancel()
    # sys.exit(0)
    os.abort()

# signal.signal(signal.SIGINT, sig_stop_handler)
# signal.signal(signal.SIGTERM, sig_stop_handler)


## ENDOF: record mem info ################################################

def port_pretrained_models(
    model_type=None,
    input_shape=None,
    num_classes=None,
):
    """
    This function loads the NN model for training

    Args:
        model_type (str, optional): type of NN model. Defaults to 'resnet50'.
        input_shape (tuple, optional): NN input shape excluding batch dim. Defaults to (224, 224, 3).
        num_classes (int, optional): number of classes of the classification task. Defaults to 1000.

    Raises:
        NotImplementedError: The requested model is not implemented

    Returns:
        tf.keras.Model: The requested NN model
    """
    
    if model_type == 'mobilenetv2':
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                       include_top=False,
                                                       weights='imagenet')
        base_model.trainable = True
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.2),
        ])
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.layers.Dense(num_classes)
        inputs = tf.keras.Input(shape=input_shape)
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = base_model(x, training=False)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        model = tf.keras.Model(inputs, outputs)
        
    elif model_type == 'resnet50':
        preprocess_input = tf.keras.applications.resnet.preprocess_input
        base_model = tf.keras.applications.ResNet50(input_shape=input_shape,
                                                    include_top=False,
                                                    weights='imagenet')
        base_model.trainable = True
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.2),
        ])
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.layers.Dense(num_classes)
        inputs = tf.keras.Input(shape=input_shape)
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = base_model(x, training=False)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        model = tf.keras.Model(inputs, outputs)
        
    elif model_type == 'vgg16':
        preprocess_input = tf.keras.applications.vgg16.preprocess_input
        base_model = tf.keras.applications.VGG16(input_shape=input_shape,
                                                 include_top=False,
                                                 weights='imagenet')
        base_model.trainable = True
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.2),
        ])
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.layers.Dense(num_classes)
        inputs = tf.keras.Input(shape=input_shape)
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = base_model(x, training=False)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        model = tf.keras.Model(inputs, outputs)
    
    elif model_type == 'vit':
        # base_model = vit.vit_b16(
        #     image_size=input_shape[0],
        #     pretrained=True,
        #     include_top=True,
        #     pretrained_top=False,
        #     weights='imagenet21k+imagenet2012',
        #     classes=num_classes,
        # )
        base_model = vit_b16(
            image_size=input_shape[0],
            pretrained=True,
            include_top=True,
            pretrained_top=False,
            weights='imagenet21k+imagenet2012',
            classes=num_classes,
        )
        base_model.trainable = True
        # base_model.layers[4].layers[:-1]
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.2),
        ])
        
        inputs = tf.keras.Input(shape=input_shape)
        x = data_augmentation(inputs)
        x = vit.preprocess_inputs(x)
        outputs = base_model(x, training=False)
        model = tf.keras.Model(inputs, outputs)
    
    else:
        raise NotImplementedError("This model has not been implemented yet")
    
    return model


def port_datasets(
    dataset_name,
    input_shape=28,
    batch_size=4,
    num_split=10,
):
    """
    This function loads the train and test splits of the requested dataset, and
    creates input pipelines for training.

    Args:
        dataset_name (str): name of the dataset
        input_shape (tuple): NN input shape excluding batch dim
        batch_size (int): batch size of training split, 
        default batch size for testing split is batch_size*2
    
    Raises:
        NotImplementedError: The requested dataset is not implemented

    Returns:
        Train and test splits of the request dataset
    """
    
    # maximize number limit of opened files
    low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))
    
    def prep(x, y):
        x = tf.image.resize(x, [input_shape[0], input_shape[1]])
        return x, y
                               
        
    if dataset_name == 'caltech_birds2011':

        splits = tfds.even_splits('train', n=num_split)
        client_datasets = []

        for split in splits:
            ds_train = tfds.load('caltech_birds2011', split=split, as_supervised=True)
            ds_train = ds_train.map(prep, num_parallel_calls=tf.data.AUTOTUNE) \
                .batch(batch_size) \
                .prefetch(buffer_size=tf.data.AUTOTUNE)
            client_datasets.append(ds_train)

        ds_test = tfds.load('caltech_birds2011', split='test', as_supervised=True)

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
          client_id = y % num_split
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

    elif dataset_name == 'cifar10-noniid':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        # 数据预处理
        x_train = x_train.astype('float32') / 255.0  # 归一化像素值
        x_test = x_test.astype('float32') / 255.0

        # 根据标签将训练数据集划分为4个子集
        client_datasets = [[] for _ in range(num_split)]
        for x, y in zip(x_train, y_train):
            client_id = y[0] // (10 // num_split)  # mnist有10个类别,平均分成4组
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


    else:
        raise NotImplementedError("This dataset has not been implemented yet")
                              
    return client_datasets, ds_test
