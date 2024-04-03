import tensorflow as tf
import numpy as np
from utils import port_datasets  # 确保能从你的环境中导入
from utils import port_pretrained_models  # 确保能从你的环境中导入
from train import elastic_training
from train import full_training


global_accuracy_ft = 0
global_accuracy_fet = 0
global_accuracy_fetc = 0

def federated_training(client_datasets, ds_test, model_type='resnet50', global_epochs=4,
                       num_classes=37):

    input_shape = (32,32,3)  
    global_model = port_pretrained_models(model_type=model_type, input_shape=input_shape,
                                          num_classes=num_classes)  

    for global_epoch in range(global_epochs):
        print(f"Global Epoch {global_epoch + 1}/{global_epochs}")
        client_weights = []

        for client_id, ds_train in enumerate(client_datasets):
            print(f"Training on client {client_id + 1}/{len(client_datasets)}")
            client_model = port_pretrained_models(model_type=model_type, input_shape=input_shape,
                                                  num_classes=num_classes)  
            client_model.set_weights(global_model.get_weights())  

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

            client_weights.append(client_model.get_weights())  

        new_weights = np.mean(client_weights, axis=0)
        global_model.set_weights(new_weights)

        global_model.compile(optimizer='sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])
        test_loss, test_accuracy = global_model.evaluate(ds_test, verbose=0)
        print(f"Global test accuracy: {test_accuracy * 100:.2f}%")
        global global_accuracy_ft  
        global_accuracy_ft = test_accuracy

    return global_model

def federated_elastic_training(client_datasets, ds_test, model_type='resnet50', global_epochs=4,
                               num_classes=10, timing_info='timing_info'):

    input_shape = (32, 32, 3)  # Preset input shape
    global_model = port_pretrained_models(model_type=model_type, input_shape=input_shape,
                                          num_classes=num_classes)  # Load global model

    for global_epoch in range(global_epochs):
        print(f"Global Epoch {global_epoch + 1}/{global_epochs}")
        client_weights = []
        client_importances = []

        for client_id, ds_train in enumerate(client_datasets):
            print(f"Training on client {client_id + 1}/{len(client_datasets)}")
            client_model = port_pretrained_models(model_type=model_type, input_shape=input_shape,
                                                  num_classes=num_classes)  # Create model for each client
            client_model.set_weights(global_model.get_weights())  # Initialize with global model weights

            I_C = elastic_training(
                model=client_model,
                model_name=model_type,
                ds_train=ds_train,
                ds_test=ds_test,
                run_name='auto',
                logdir='logs',
                timing_info=timing_info,
                optim='sgd',
                lr=1e-4,
                weight_decay=5e-4,
                epochs=5,
                interval=5,
                rho=0.533,
                disable_random_id=True,
                save_model=False,
                save_txt=False
            )

            client_weights.append(client_model.get_weights())  # Collect trained weights
            client_importances.append(I_C)  # Collect client importance measures

        if global_epoch == 0:
            # In the first global epoch, use FedAvg to get global model weights
            new_weights = np.mean(client_weights, axis=0)
        else:
            # From the second global epoch onwards
            # Compute global model importance measure
            global_weights_before = global_model.get_weights()
            global_model.set_weights(new_weights)
            global_weights_after = global_model.get_weights()
            global_weight_delta = [after - before for after, before in zip(global_weights_after, global_weights_before)]

            for x_probe, y_probe in ds_test.take(1):
                with tf.GradientTape() as tape:
                    y_pred = global_model(x_probe, training=True)
                    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_probe, y_pred)
                global_gradients_after = tape.gradient(loss, global_model.trainable_variables)

            I_g = [tf.reduce_sum(grad * delta) for grad, delta in zip(global_gradients_after, global_weight_delta)]
            I_g = tf.convert_to_tensor(I_g)
            I_g = I_g / tf.reduce_max(tf.abs(I_g))

            # Update client importance measures with weighted sum of client and global importance
            updated_client_importances = [0.4 * I_C + 0.6 * I_g for I_C in client_importances]

            # Compute new global model weights using FedAvg with updated client importance measures
            weighted_client_weights = [importance * weight for importance, weight in
                                       zip(updated_client_importances, client_weights)]
            new_weights = np.sum(weighted_client_weights, axis=0) / np.sum(updated_client_importances)

        global_model.set_weights(new_weights)

        global_model.compile(optimizer='sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                             metrics=['accuracy'])
        test_loss, test_accuracy = global_model.evaluate(ds_test, verbose=0)
        print(f"Global test accuracy: {test_accuracy * 100:.2f}%")
        global global_accuracy_fet
        global_accuracy_fet = test_accuracy

    return global_model




if __name__ == '__main__':

    dataset_name = 'mnist'
    model_type = 'resnet50'
    model_name = 'resnet50'
    num_classes = 10  
    batch_size = 4
    input_shape = (32,32,3)
    timing_info = model_name + '_' + str(input_shape) + '_' + str(num_classes) + '_' + str(batch_size) + '_' + 'profile'
    client_datasets, ds_test = port_datasets(dataset_name, (32,32,3), batch_size)

    federated_training(client_datasets, ds_test, model_type=model_type, num_classes=num_classes)
    federated_elastic_training(client_datasets, ds_test, model_type='resnet50', global_epochs=4,
                               num_classes=10, timing_info=timing_info)
    
    print(f"Federated Training Accuracy: {global_accuracy_ft * 100:.2f}%")
    print(f"Federated Elastic Training Accuracy: {global_accuracy_fet * 100:.2f}%")
