import tensorflow as tf
import numpy as np
from utils import port_datasets  # 确保能从你的环境中导入
from utils import port_pretrained_models  # 确保能从你的环境中导入
from train import elastic_training
from train import full_training


def federated_training(client_datasets, ds_test, model_type='resnet50', global_epochs=4,
                       num_classes=37):
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

            full_training(
                model=client_model,
                ds_train=ds_train,
                ds_test=ds_test,
                run_name='auto',
                logdir='logs',
                optim='sgd',
                lr=1e-4,
                weight_decay=5e-4,
                epochs=5,
                disable_random_id=True,
                save_model=False,
                save_txt=False
            )

            client_weights.append(client_model.get_weights())  # 收集训练后的权重

        # 使用FedAvg算法更新全局模型的权重
        new_weights = np.mean(client_weights, axis=0)
        global_model.set_weights(new_weights)

    # 在全局测试集上评估全局模型
        global_model.compile(optimizer='sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])
        test_loss, test_accuracy = global_model.evaluate(ds_test, verbose=0)
        print(f"Global test accuracy: {test_accuracy * 100:.2f}%")

    return global_model

def federated_elastic_training(client_datasets, ds_test, model_type='resnet50', global_epochs=4,
                               num_classes=37, timing_info='timing_info'):

    input_shape = (224, 224, 3)  # Preset input shape
    global_model = port_pretrained_models(model_type=model_type, input_shape=input_shape,
                                          num_classes=num_classes)  # Load global model

    for global_epoch in range(global_epochs):
        print(f"Global Epoch {global_epoch + 1}/{global_epochs}")
        client_weights = []

        for client_id, ds_train in enumerate(client_datasets):
            print(f"Training on client {client_id + 1}/{len(client_datasets)}")
            client_model = port_pretrained_models(model_type=model_type, input_shape=input_shape,
                                                  num_classes=num_classes)  # Create model for each client
            client_model.set_weights(global_model.get_weights())  # Initialize with global model weights

            elastic_training(
                model=client_model,
                model_name=model_type,  # Assuming you have a way to infer the model name from the type
                ds_train=ds_train,
                ds_test=ds_test,
                run_name='auto',
                logdir='logs',
                timing_info=timing_info,
                optim='sgd',
                lr=1e-4,
                weight_decay=5e-4,
                epochs=5,
                interval=1,  # Setting to 1 for federated learning scenario
                rho=0.533,
                disable_random_id=True,
                save_model=False,
                save_txt=False
            )

            client_weights.append(client_model.get_weights())  # Collect trained weights

        # Update global model weights using FedAvg algorithm
        new_weights = np.mean(client_weights, axis=0)
        global_model.set_weights(new_weights)

    # Evaluate global model on global test set
    global_model.compile(optimizer='sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])
    test_loss, test_accuracy = global_model.evaluate(ds_test, verbose=0)
    print(f"Global test accuracy: {test_accuracy * 100:.2f}%")

    return global_model

def federated_elastic_training_compare(client_datasets, ds_test, model_type='resnet50', global_epochs=4,
                               num_classes=37, timing_info='timing_info'):
    input_shape = (224, 224, 3)
    global_model = port_pretrained_models(model_type=model_type, input_shape=input_shape,
                                          num_classes=num_classes)

    for global_epoch in range(global_epochs):
        print(f"Global Epoch  {global_epoch + 1}/{global_epochs}")
        client_weights = []

        for client_id, ds_train in enumerate(client_datasets):
            print(f"Training on client {client_id + 1}/{len(client_datasets)}")
            client_model = port_pretrained_models(model_type=model_type, input_shape=input_shape,
                                                  num_classes=num_classes)
            client_model.set_weights(global_model.get_weights())

            elastic_training(
                model=client_model,
                model_name=model_type,  # Assuming you have a way to infer the model name from the type
                ds_train=ds_train,
                ds_test=ds_test,
                run_name='auto',
                logdir='logs',
                timing_info=timing_info,
                optim='sgd',
                lr=1e-4,
                weight_decay=5e-4,
                epochs=5,
                interval=1,  # Setting to 1 for federated learning scenario
                rho=0.533,
                disable_random_id=True,
                save_model=False,
                save_txt=False
            )

            client_weights.append(client_model.get_weights())  # 收集训练后的权重

        # 对每个客户端权重进行条件更新
        global_weights = global_model.get_weights()
        updated_client_weights = []
        for client_weight in client_weights:
            updated_weights = []
            for global_w, client_w in zip(global_weights, client_weight):
                if np.any(client_w < global_w):
                    updated_w = client_w * 0.4 + global_w * 0.6
                else:
                    updated_w = client_w
                updated_weights.append(updated_w)
            updated_client_weights.append(updated_weights)

        # 计算更新后的平均权重，并设置为全局模型的新权重
        new_weights = np.mean(updated_client_weights, axis=0)
        global_model.set_weights(new_weights)

    # 在全局测试集上评估全局模型
    global_model.compile(optimizer='sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])
    test_loss, test_accuracy = global_model.evaluate(ds_test, verbose=0)
    print(f"Global test accuracy: {test_accuracy * 100:.2f}%")

    return global_model

if __name__ == '__main__':
    # 设置数据集名称和模型参数
    dataset_name = 'oxford_iiit_pet'
    model_type = 'resnet50'
    model_name = 'resnet50'
    num_classes = 37  # Oxford IIIT Pet
    input_size = 224
    batch_size = 4
    timing_info = model_name + '_' + str(input_size) + '_' + str(num_classes) + '_' + str(batch_size) + '_' + 'profile'

    # 调用port_datasets函数加载数据
    client_datasets, ds_test = port_datasets(dataset_name, (224, 224, 3), batch_size)
#    federated_training(client_datasets, ds_test, model_type=model_type, num_classes=num_classes)
    federated_elastic_training_compare(client_datasets, ds_test, model_type='resnet50', global_epochs=4,
                               num_classes=37, timing_info=timing_info)