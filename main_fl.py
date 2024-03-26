import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils import port_datasets  # 确保能从你的环境中导入
from utils import port_pretrained_models  # 确保能从你的环境中导入


def federated_training(client_datasets, ds_test, model_type='resnet50', global_epochs=4, client_epochs=5,
                       num_classes=1000):
    """模拟联邦学习环境，使用预训练模型"""
    input_shape = (224, 224, 3)  # 预设输入形状
    global_model = port_pretrained_models(model_type=model_type, input_shape=input_shape,
                                          num_classes=num_classes)  # 加载全局模型

    for global_epoch in range(global_epochs):
        print(f"Global Epoch {global_epoch + 1}/{global_epochs}")
        client_weights = []

        for client_id, ds_train in enumerate(client_datasets):
            print(f"Training on client {client_id + 1}/{len(client_datasets)}")
            client_model = port_pretrained_models(model_type=model_type, input_shape=input_shape,
                                                  num_classes=num_classes)  # 为每个客户端创建模型
            client_model.set_weights(global_model.get_weights())  # 初始化为全局模型的权重

            client_model.compile(optimizer='sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                 metrics=['accuracy'])
            client_model.fit(ds_train, epochs=client_epochs, verbose=1)
            client_weights.append(client_model.get_weights())  # 收集训练后的权重

        # 使用FedAvg算法更新全局模型的权重
        new_weights = np.mean(client_weights, axis=0)
        global_model.set_weights(new_weights)

    # 在全局测试集上评估全局模型
    global_model.compile(optimizer='sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])
    test_loss, test_accuracy = global_model.evaluate(ds_test, verbose=0)
    print(f"Global test accuracy: {test_accuracy * 100:.2f}%")


if __name__ == '__main__':
    # 设置数据集名称和模型参数
    dataset_name = 'oxford_iiit_pet'
    model_type = 'resnet50'
    num_classes = 37  # Oxford IIIT Pet 数据集有37个类别
    batch_size = 4

    # 调用port_datasets函数加载数据
    client_datasets, ds_test = port_datasets(dataset_name, (224, 224, 3), batch_size)

    # 模拟联邦学习环境
    federated_training(client_datasets, ds_test, model_type=model_type, num_classes=num_classes)
