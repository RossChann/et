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
import keras as keras
import socket
import pickle

def receive_weights(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        conn, addr = s.accept()
        with conn:
            data = b''
            while True:
                packet = conn.recv(4096)
                if not packet:
                    break
                data += packet
            weights = pickle.loads(data)
    return weights

def send_weights(weights, host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        data = pickle.dumps(weights)
        s.sendall(data)

def aggregate_weights(client_weights):
    avg_weight = [np.mean(np.array(weights), axis=0) for weights in zip(*client_weights)]
    return avg_weight

def update_global_model(global_model, aggregated_weights):
    global_model.set_weights(aggregated_weights)

def show_results(global_model, ds_test):
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

def full_training(
        model,
        ds_train,
        ds_test,
        run_name,
        logdir,
        optim='sgd',
        lr=1e-4,
        weight_decay=5e-4,
        epochs=12,
        disable_random_id=False,
        save_model=False,
        save_txt=False,
):
    """All NN weights will be trained"""

    if optim == 'sgd':
        decay_steps = len(tfds.as_numpy(ds_train)) * epochs

        lr_schedule = tf.keras.experimental.CosineDecay(lr, decay_steps=decay_steps)
        wd_schedule = tf.keras.experimental.CosineDecay(lr * weight_decay, decay_steps=decay_steps)
        optimizer = tfa.optimizers.SGDW(learning_rate=lr_schedule, weight_decay=wd_schedule, momentum=0.9,
                                        nesterov=False)
    else:
        optimizer = tf.keras.optimizers.Adam(lr)

    loss_fn_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    if disable_random_id:
        runid = run_name
    else:
        runid = run_name + '_full_x' + str(np.random.randint(10000))
    writer = tf.summary.create_file_writer(logdir + '/' + runid)
    accuracy = tf.metrics.SparseCategoricalAccuracy()
    cls_loss = tf.metrics.Mean()

    print(f"RUNID: {runid}")

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = loss_fn_cls(y, y_pred)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        accuracy(y, y_pred)
        cls_loss(loss)

    @tf.function
    def test_step(x, y):
        y_pred = model(x, training=False)
        loss = loss_fn_cls(y, y_pred)
        accuracy(y, y_pred)
        cls_loss(loss)

    training_step = 0
    best_validation_acc = 0

    clear_cache_and_rec_usage()

    total_time_0 = 0
    total_time_1 = 0
    for epoch in range(epochs):

        t0 = time.time()
        for x, y in tqdm(ds_train, desc=f'epoch {epoch + 1}/{epochs}', ascii=True):

            training_step += 1

            train_step(x, y)

            if training_step % 200 == 0:
                with writer.as_default():
                    c_loss, acc = cls_loss.result(), accuracy.result()
                    tf.summary.scalar('train/accuracy', acc, training_step)
                    tf.summary.scalar('train/classification_loss', c_loss, training_step)
                    tf.summary.scalar('train/learnig_rate', optimizer._decayed_lr('float32'), training_step)
                    cls_loss.reset_states()
                    accuracy.reset_states()
                clear_cache_and_rec_usage()

        cls_loss.reset_states()
        accuracy.reset_states()

        t1 = time.time()
        print("per epoch time(s) excluding validation:", t1 - t0)
        total_time_0 += (t1 - t0)

        for x, y in ds_test:
            test_step(x, y)

        with writer.as_default():
            tf.summary.scalar('test/classification_loss', cls_loss.result(), step=training_step)
            tf.summary.scalar('test/accuracy', accuracy.result(), step=training_step)

            if accuracy.result() > best_validation_acc:
                best_validation_acc = accuracy.result()
                if save_model:
                    model.save_weights(os.path.join('saved_models', runid + '.tf'))
                print("=================================")
                print("acc: ", accuracy.result())
                print("=================================")

            cls_loss.reset_states()
            accuracy.reset_states()

        t2 = time.time()
        print("per epoch time(s) including validation:", t2 - t0)
        total_time_1 += (t2 - t0)

        clear_cache_and_rec_usage()

    # print("total time excluding validation (s):", total_time_0)
    # print("total time including validation (s):", total_time_1)
    best_validation_acc = best_validation_acc.numpy() * 100
    total_time_0 /= 3600
    print('===============================================')
    print('Training Type: Full training')
    print(f"Accuracy (%): {best_validation_acc:.2f}")
    print(f"Time (h): {total_time_0:.2f}")
    print('===============================================')
    if save_txt:
        np.savetxt(logdir + '/' + runid + '.txt', np.array([total_time_0, best_validation_acc]))
    # sig_stop_handler(None, None)





def federated_elastic_training_server(client_datasets, ds_test, model_type='resnet50', global_epochs=4,
                                      num_classes=37, timing_info='timing_info', lr=1e-4, weight_decay=5e-4):
    global_model = port_pretrained_models(model_type=model_type, input_shape=input_shape,
                                          num_classes=num_classes)  # Load global model

    for global_epoch in range(global_epochs):
        client_host = '192.168.1.243'
        client_port = '8001'
        server_host = '192.168.1.248'
        server_port = '8000'
        print(f"Global Epoch {global_epoch + 1}/{global_epochs}")
        client_weights = []

        print("Training on device 1 (server)")
        device1_model = port_pretrained_models(model_type=model_type, input_shape=input_shape,
                                               num_classes=num_classes)
        device1_model.set_weights(global_model.get_weights())
        device1_model = full_training(device1_model, client_datasets[0], ds_test, run_name='auto', logdir='auto',
                                      optim='sgd', lr=1e-4, weight_decay=5e-4, epochs=12, disable_random_id=False,
                                      save_model=False, save_txt=False)
        client_weights.append(device1_model.get_weights())

        send_weights(global_model.get_weights(), client_host, client_port)  # 发送全局模型权重给客户端
        client_weights.append(receive_weights(server_host, server_port))  # 接收客户端更新后的权重

        aggregated_weights = aggregate_weights(client_weights)
        update_global_model(global_model, aggregated_weights)

        show_results(global_model, ds_test)

    return global_model

if __name__ == '__main__':
    dataset_name = 'oxford_iiit_pet'
    model_type = 'resnet50'
    model_name = 'resnet50'
    num_classes = 37
    global_epochs = 4
    batch_size = 4
    input_size = 224
    input_shape = (input_size, input_size, 3)
    timing_info = model_name + '_' + str(input_size) + '_' + str(num_classes) + '_' + str(batch_size) + '_' + 'profile'

    # port datasets
    client_datasets, ds_test = port_datasets(dataset_name, input_shape, batch_size)

    # train
    global_model = federated_elastic_training_server(client_datasets[0], ds_test, model_type=model_type,
                                                     global_epochs=global_epochs,
                                                     num_classes=num_classes, timing_info=timing_info)