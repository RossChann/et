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

global_accuracy_ft = 0
global_accuracy_fet = 0
global_accuracy_fetc = 0

def full_training(
        model,
        ds_train,
        ds_test,
        run_name,
        logdir,
        optim='sgd',
        lr=1e-4,
        weight_decay=5e-4,
        epochs=5,
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


def elastic_training(
        model,
        model_name,
        ds_train,
        ds_test,
        run_name,
        logdir,
        timing_info,
        optim='sgd',
        lr=1e-4,
        weight_decay=5e-4,
        epochs=5,
        interval=5,
        rho=0.533,
        disable_random_id=False,
        save_model=False,
        save_txt=False,
):
    """Train with ElasticTrainer"""

    def rho_for_backward_pass(rho):
        return (rho - 1 / 3) * 3 / 2

    t_dw, t_dy = profile_parser(
        model,
        model_name,
        5,
        'profile_extracted/' + timing_info,
        draw_figure=False,
    )
    # np.savetxt('t_dy.out', t_dy)
    # np.savetxt('t_dw.out', t_dw)
    t_dy_q, t_dw_q, disco = downscale_t_dy_and_t_dw(t_dy, t_dw, Tq=1e3)
    t_dy_q = np.flip(t_dy_q)
    t_dw_q = np.flip(t_dw_q)

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
        runid = run_name + '_elastic_x' + str(np.random.randint(10000))

    writer = tf.summary.create_file_writer(logdir + '/' + runid)
    accuracy = tf.metrics.SparseCategoricalAccuracy()
    cls_loss = tf.metrics.Mean()

    print(f"RUNID: {runid}")

    var_list = []
    # initialze a gradient list in FL
    gradients_list = []

    def train_step(x, y):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = loss_fn_cls(y, y_pred)
        gradients = tape.gradient(loss, var_list)
        optimizer.apply_gradients(zip(gradients, var_list))
        accuracy(y, y_pred)
        cls_loss(loss)

    @tf.function
    def test_step(x, y):
        y_pred = model(x, training=False)
        loss = loss_fn_cls(y, y_pred)
        accuracy(y, y_pred)
        cls_loss(loss)

    @tf.function
    def compute_dw(x, y):
        with tf.GradientTape() as tape:
            y_pred_0 = model(x, training=True)
            loss_0 = loss_fn_cls(y, y_pred_0)
        grad_0 = tape.gradient(loss_0, model.trainable_weights)
        w_0 = [w.value() for w in model.trainable_weights]  # record initial weight values
        optimizer.apply_gradients(zip(grad_0, model.trainable_weights))
        w_1 = [w.value() for w in model.trainable_weights]  # record weight values after applying optimizer
        dw_0 = [w_1_k - w_0_k for (w_0_k, w_1_k) in zip(w_0, w_1)]  # compute weight changes
        with tf.GradientTape() as tape:
            y_pred_1 = model(x, training=True)
            loss_1 = loss_fn_cls(y, y_pred_1)
        grad_1 = tape.gradient(loss_1, model.trainable_weights)
        I = [tf.reduce_sum((grad_1_k * dw_0_k)) for (grad_1_k, dw_0_k) in zip(grad_1, dw_0)]
        I = tf.convert_to_tensor(I)
        I = I / tf.reduce_max(tf.abs(I))
        # restore weights
        for k, w in enumerate(model.trainable_weights):
            w.assign(w_0[k])
        return dw_0, I

    training_step = 0
    best_validation_acc = 0

    total_time_0 = 0
    total_time_1 = 0
    for epoch in range(epochs):

        t0 = time.time()
        if epoch % interval == 0:
            for x_probe, y_probe in ds_train.take(1):
                dw, I = compute_dw(x_probe, y_probe)
                I = -I.numpy()
                I = np.flip(I)
                # np.savetxt('importance.out', I)
                rho_b = rho_for_backward_pass(rho)
                max_importance, m = selection_DP(t_dy_q, t_dw_q, I, rho=rho_b * disco)
                m = np.flip(m)
                print("m:", m)
                print("max importance:", max_importance)
                print("%T_sel:", 100 * np.sum(np.maximum.accumulate(m) * t_dy + m * t_dw) / np.sum(t_dy + t_dw))
                var_list = []
                all_vars = model.trainable_weights
                for k, m_k in enumerate(m):
                    if tf.equal(m_k, 1):
                        var_list.append(all_vars[k])
                train_step_cpl = tf.function(train_step)
        gradients_list = []
        for x, y in tqdm(ds_train, desc=f'epoch {epoch + 1}/{epochs}', ascii=True):

            training_step += 1
            train_step_cpl(x, y)

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

    best_validation_acc = best_validation_acc.numpy() * 100
    total_time_0 /= 3600
    print('===============================================')
    print('Training Type: ElasticTrainer')
    print(f"Accuracy (%): {best_validation_acc:.2f}")
    print(f"Time (h): {total_time_0:.2f}")
    print('===============================================')
    if save_txt:
        np.savetxt(logdir + '/' + runid + '.txt', np.array([total_time_0, best_validation_acc]))

    return model


def elastic_training_updated(
        I_G,
        model,
        model_name,
        ds_train,
        ds_test,
        run_name,
        logdir,
        timing_info,
        optim='sgd',
        lr=1e-4,
        weight_decay=5e-4,
        epochs=5,
        interval=5,
        rho=0.533,
        disable_random_id=False,
        save_model=False,
        save_txt=False,

):
    """Train with ElasticTrainer"""

    def rho_for_backward_pass(rho):
        return (rho - 1 / 3) * 3 / 2

    t_dw, t_dy = profile_parser(
        model,
        model_name,
        5,
        'profile_extracted/' + timing_info,
        draw_figure=False,
    )
    # np.savetxt('t_dy.out', t_dy)
    # np.savetxt('t_dw.out', t_dw)
    t_dy_q, t_dw_q, disco = downscale_t_dy_and_t_dw(t_dy, t_dw, Tq=1e3)
    t_dy_q = np.flip(t_dy_q)
    t_dw_q = np.flip(t_dw_q)

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
        runid = run_name + '_elastic_x' + str(np.random.randint(10000))

    writer = tf.summary.create_file_writer(logdir + '/' + runid)
    accuracy = tf.metrics.SparseCategoricalAccuracy()
    cls_loss = tf.metrics.Mean()

    print(f"RUNID: {runid}")

    var_list = []
    # initialze a gradient list in FL
    gradients_list = []

    def train_step(x, y):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = loss_fn_cls(y, y_pred)
        gradients = tape.gradient(loss, var_list)
        optimizer.apply_gradients(zip(gradients, var_list))
        accuracy(y, y_pred)
        cls_loss(loss)

    @tf.function
    def test_step(x, y):
        y_pred = model(x, training=False)
        loss = loss_fn_cls(y, y_pred)
        accuracy(y, y_pred)
        cls_loss(loss)

    @tf.function
    def compute_dw(x, y, I_G):
        with tf.GradientTape() as tape:
            y_pred_0 = model(x, training=True)
            loss_0 = loss_fn_cls(y, y_pred_0)
        grad_0 = tape.gradient(loss_0, model.trainable_weights)
        w_0 = [w.value() for w in model.trainable_weights]  # record initial weight values
        optimizer.apply_gradients(zip(grad_0, model.trainable_weights))
        w_1 = [w.value() for w in model.trainable_weights]  # record weight values after applying optimizer
        dw_0 = [w_1_k - w_0_k for (w_0_k, w_1_k) in zip(w_0, w_1)]  # compute weight changes
        with tf.GradientTape() as tape:
            y_pred_1 = model(x, training=True)
            loss_1 = loss_fn_cls(y, y_pred_1)
        grad_1 = tape.gradient(loss_1, model.trainable_weights)
        I = [tf.reduce_sum((grad_1_k * dw_0_k)) for (grad_1_k, dw_0_k) in zip(grad_1, dw_0)]
        I = tf.convert_to_tensor(I)
        I = I / tf.reduce_max(tf.abs(I))

        # Compute the weighted average of I and I_G_selected
        alpha = 0.6  # Weight for I_G_selected
        I_weighted = alpha * I_G + (1 - alpha) * I
        # restore weights
        for k, w in enumerate(model.trainable_weights):
            w.assign(w_0[k])
        return dw_0, I_weighted

    training_step = 0
    best_validation_acc = 0

    total_time_0 = 0
    total_time_1 = 0
    for epoch in range(epochs):

        t0 = time.time()
        if epoch % interval == 0:
            for x_probe, y_probe in ds_train.take(1):
                dw, I = compute_dw(x_probe, y_probe, I_G)
                I = -I.numpy()
                I = np.flip(I)
                # np.savetxt('importance.out', I)
                rho_b = rho_for_backward_pass(rho)
                max_importance, m = selection_DP(t_dy_q, t_dw_q, I, rho=rho_b * disco)
                m = np.flip(m)
                print("m:", m)
                print("max importance:", max_importance)
                print("%T_sel:", 100 * np.sum(np.maximum.accumulate(m) * t_dy + m * t_dw) / np.sum(t_dy + t_dw))
                var_list = []
                all_vars = model.trainable_weights
                for k, m_k in enumerate(m):
                    if tf.equal(m_k, 1):
                        var_list.append(all_vars[k])
                train_step_cpl = tf.function(train_step)
        gradients_list = []
        for x, y in tqdm(ds_train, desc=f'epoch {epoch + 1}/{epochs}', ascii=True):

            training_step += 1
            train_step_cpl(x, y)

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

    best_validation_acc = best_validation_acc.numpy() * 100
    total_time_0 /= 3600
    print('===============================================')
    print('Training Type: ElasticTrainer')
    print(f"Accuracy (%): {best_validation_acc:.2f}")
    print(f"Time (h): {total_time_0:.2f}")
    print('===============================================')
    if save_txt:
        np.savetxt(logdir + '/' + runid + '.txt', np.array([total_time_0, best_validation_acc]))

    return model


def federated_elastic_training_advanced(client_datasets, ds_test, model_type='resnet50', global_epochs=4,
                                        num_classes=37, timing_info='timing_info', lr=1e-4, weight_decay=5e-4):
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

    #######################
    global_model = port_pretrained_models(model_type=model_type, input_shape=input_shape,
                                          num_classes=num_classes)  # Load global model

    for global_epoch in range(global_epochs):
        print(f"Global Epoch {global_epoch + 1}/{global_epochs}")
        client_weights = []
        compute_client_weights = []
        if global_epoch == 0:
            for client_id, ds_train in enumerate(client_datasets):
                print(f"Training on client {client_id + 1}/{len(client_datasets)}")
                client_model = port_pretrained_models(model_type=model_type, input_shape=input_shape,
                                                      num_classes=num_classes)  # Create model for each client and initailze the weights
                client_model=full_training(client_model, model_name, ds_train, ds_test, run_name='auto',
                                                logdir='auto', timing_info=timing_info, optim='sgd', lr=1e-4,
                                                weight_decay=5e-4, epochs=5, rho=0.533,
                                                disable_random_id=True, save_model=False, save_txt=False)
                client_model = elastic_training(client_model, model_name, ds_train, ds_test, run_name='auto',
                                                logdir='auto', timing_info=timing_info, optim='sgd', lr=1e-4,
                                                weight_decay=5e-4, epochs=5, interval=5, rho=0.533,
                                                disable_random_id=True, save_model=False, save_txt=False)  # train

                client_model=full_training(client_model,ds_train,ds_test,run_name='auto',logdir='auto',optim='sgd',lr=1e-4,weight_decay=5e-4,epochs=5)

                compute_client_weights.append(client_model.get_weights())  # 320
            compute_weights = aggregate_weights(compute_client_weights)  # 320

            update_global_model(global_model, compute_weights)  # 320->320

            show_results(global_model)

        else:
            for client_id, ds_train in enumerate(client_datasets):
                print(f"Training on client {client_id + 1}/{len(client_datasets)}")
                client_model.set_weights(global_model.get_weights())
                client_model = full_training(client_model,ds_train,ds_test,run_name='auto',logdir='auto',optim='sgd',lr=1e-4,weight_decay=5e-4,epochs=5)
                compute_client_weights.append(client_model.get_weights())  # 320
            compute_weights = aggregate_weights(compute_client_weights)  # 320

            update_global_model(global_model, compute_weights)  # 320->320


            show_results(global_model)

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
    global_model = federated_elastic_training_advanced(client_datasets, ds_test, model_type=model_type,
                                                       global_epochs=global_epochs,
                                                       num_classes=num_classes, timing_info=timing_info)

