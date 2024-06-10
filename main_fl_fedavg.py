import tensorflow as tf
import numpy as np
from utils import port_datasets
from utils import port_pretrained_models
from selection_solver_DP import selection_DP, downscale_t_dy_and_t_dw
from profiler import profile_parser
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import time
from tqdm import tqdm
import os
from utils import clear_cache_and_rec_usage
from tensorflow.keras import backend as K
from train import elastic_training
import keras as keras
from train import elastic_training_pcandorin
from train import full_training
global_accuracy_ft = 0
global_accuracy_fet = 0
global_accuracy_fetc = 0





def federated_elastic_training_advanced(client_datasets, ds_test, model_type='vgg16', global_epochs=20,
                                        num_classes=10, timing_info='timing_info', lr=1e-4, weight_decay=5e-4):
    #######################
    def aggregate_weights(client_weights):
        avg_weight = [np.mean(np.array(weights), axis=0) for weights in zip(*client_weights)]
        return avg_weight

    def update_global_model(global_model, aggregated_weights):
        global_model.set_weights(aggregated_weights)

    def show_results(global_model):
        loss_fn_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        accuracy = tf.metrics.SparseCategoricalAccuracy()
        cls_loss = tf.metrics.Mean()

        for x, y in ds_test:
            y_pred = global_model(x, training=False)
            loss = loss_fn_cls(y, y_pred)
            accuracy(y, y_pred)
            cls_loss(loss)

        print('===============================================')
        print(f"Global Model Accuracy (%): {accuracy.result().numpy() * 100:.2f}")
        print('===============================================')

        with open('global_accuracy.txt', 'a') as file:
            file.write(f"{accuracy.result().numpy() * 100:.2f}%\n")
    #######################
    global_model = port_pretrained_models(model_type=model_type, input_shape=input_shape,
                                          num_classes=num_classes)  # Load global model

    for global_epoch in range(global_epochs):
        print(f"Global Epoch {global_epoch + 1}/{global_epochs}")
        compute_client_weights = []
        if global_epoch == 0:
            for client_id, ds_train in enumerate(client_datasets):
                print(f"Training on client {client_id + 1}/{len(client_datasets)}")
                client_model = port_pretrained_models(model_type=model_type, input_shape=input_shape,
                                                      num_classes=10)  # Create model for each client and initailze the weights

                client_model = elastic_training_pcandorin(client_model, model_name, ds_train, ds_test, run_name='auto',
                                                          logdir='auto', timing_info=timing_info, optim='sgd', lr=1e-4,
                                                          weight_decay=5e-4, epochs=5, interval=5, rho=0.533,
                                                          disable_random_id=True, save_model=False, save_txt=False,
                                                          id=client_id)  # train
                client_model = full_training(client_model, ds_train, ds_test, run_name='auto', logdir='auto',
                                             optim='sgd', lr=1e-4, weight_decay=5e-4, epochs=5, disable_random_id=True,
                                             save_model=False, save_txt=False)
                compute_client_weights.append(client_model.get_weights())  # 320
            compute_weights = aggregate_weights(compute_client_weights)  # 320
            w_0 = K.batch_get_value(global_model.trainable_weights)  # 320->214
            update_global_model(global_model, compute_weights)  # 320->320
            w_1 = K.batch_get_value(global_model.trainable_weights)  # 320-214
            show_results(global_model)

        else:
            for client_id, ds_train in enumerate(client_datasets):
                print(f"Training on client {client_id + 1}/{len(client_datasets)}")
                client_model.set_weights(global_model.get_weights())
                client_model = elastic_training_pcandorin(client_model, model_name, ds_train, ds_test, run_name='auto',
                                                          logdir='auto', timing_info=timing_info, optim='sgd', lr=1e-4,
                                                          weight_decay=5e-4, epochs=5, interval=5, rho=0.533,
                                                          disable_random_id=True, save_model=False, save_txt=False,
                                                          id=client_id)  # train
                client_model = full_training(client_model, ds_train, ds_test, run_name='auto', logdir='auto',
                                             optim='sgd', lr=1e-4, weight_decay=5e-4, epochs=5, disable_random_id=True,
                                             save_model=False, save_txt=False)
                compute_client_weights.append(client_model.get_weights())  # 320
            compute_weights = aggregate_weights(compute_client_weights)  # 320
            w_0 = K.batch_get_value(global_model.trainable_weights)  # 320->214
            update_global_model(global_model, compute_weights)  # 320->320
            w_1 = K.batch_get_value(global_model.trainable_weights)  # 320-214

            show_results(global_model)

    return global_model


if __name__ == '__main__':
    dataset_name = 'mnist-dirichlet'
    model_type = 'resnet50'
    model_name = 'resnet50'
    num_classes = 10
    global_epochs = 20
    batch_size = 4
    input_size = 32
    input_shape = (input_size, input_size, 3)
    timing_info = model_name + '_' + str(input_size) + '_' + str(num_classes) + '_' + str(batch_size) + '_' + 'profile'

    # port datasets
    client_datasets, ds_test = port_datasets(dataset_name, input_shape, batch_size)

    # train
    global_model = federated_elastic_training_advanced(client_datasets, ds_test, model_type=model_type,
                                                       global_epochs=global_epochs,
                                                       num_classes=num_classes, timing_info=timing_info)

