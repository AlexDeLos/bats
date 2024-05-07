from pathlib import Path
# import tensorflow as tf
import cupy as cp
import numpy as np
import sys
import os
import sys



sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# from Dataset import Dataset
from Dataset_cifar import Dataset

from bats.Monitors import *
from bats.Layers import LIFLayer
from bats.Losses import *
from bats.Network import Network
from bats.Optimizers import *
from bats.Layers.ConvInputLayer import ConvInputLayer
from bats.Layers.ConvLIFLayer import ConvLIFLayer

from bats.Layers.ConvLIFLayer_new_Residual import ConvLIFLayer_new_Residual
from bats.Utils.utils import get_arguments

from bats.Layers.PoolingLayer import PoolingLayer
from experiments.utils.utils import build_network_SCNN, wandb_handler


arguments = get_arguments()
# Residual arguments
N_HIDDEN_LAYERS = arguments.n_hidden_layers
USE_RESIDUAL = arguments.use_residual
RESIDUAL_EVERY_N = arguments.residual_every_n
FULL_METRIC = False

# Change from small test on computer to big test on cluster
CLUSTER = arguments.cluster
STANDARD = arguments.standard
USE_WANDB = arguments.use_wanb
ALTERNATE = arguments.alternate
USE_RESIDUAL = arguments.use_residual
RESIDUAL_JUMP_LENGTH = arguments.residual_jump_length
FIX_SEED = False
USE_PADDING = arguments.use_pad #! residual and padd gives nans
USE_DELAY = arguments.use_delay
TTFS = arguments.ttfs
RESTORE = arguments.restore
USE_CIFAR100 = arguments.cifar100	
USE_COURSE_LABELS = arguments.use_coarse_labels
USE_3_CHANNELS = arguments.use_3_channels #! false could be broken
#TODO: try to get the non append function to run out of memory
if USE_CIFAR100:
    DATASET_PATH = "./datasets/cifar-100-python/"
else:
    DATASET_PATH = "./datasets/cifar-10-batches-py/"
# what causes nans:
#! residual layers with pre = jump and nans
# Why is it not learning?

# but silent labels go down and kind of does loss
#TODO: try to get the non append function to run out of memory

NUMBER_OF_RUNS = arguments.runs


if USE_3_CHANNELS:
    INPUT_SHAPE = np.array([32, 32, 3])
else:
    INPUT_SHAPE = np.array([32, 32, 1])
# INPUT_SHAPE = np.array([5,5,2])
SIMULATION_TIME = 0.2
CHANNELS = 24  # we should use 64...
conv_var = {
    'filter': np.array([3, 3, CHANNELS]),
    'tau_s': 0.130,
    'threshold_hat': 0.15,
    'delta_threshold': 1 * 0.15,
    'spike_buffer_size': 5
}
conv_res_var = {
    'filter': np.array([3, 3, CHANNELS]),
    'tau_s': 0.130,
    'threshold_hat': 0.1,
    'delta_threshold': 1 * 0.1,
    'spike_buffer_size': 5
}
fc_var = {
    'n_neurons': 500,
    'tau_s': 0.130,
    'threshold_hat': 0.5,
    'delta_threshold': 1 * 0.5,
    'spike_buffer_size': 5
}
# Output_layer
if USE_COURSE_LABELS and USE_CIFAR100:
    N_OUTPUTS = 20
elif USE_CIFAR100:
    N_OUTPUTS = 100
else:
    N_OUTPUTS = 10

if TTFS:
    out_buffer_size = 1
else:
    out_buffer_size = 20
output_var = {
    'n_neurons': N_OUTPUTS,
    'tau_s': 0.130,
    'threshold_hat': 0.5,
    'delta_threshold': 1 * 0.5,
    'spike_buffer_size': out_buffer_size
}

N_TRAINING_EPOCHS = arguments.n_epochs
TRAIN_BATCH_SIZE = arguments.batch_size #! used to be 50 -> putting it at 50 crashes the cluster when using append
TEST_BATCH_SIZE = arguments.batch_size
if CLUSTER:
    N_TRAIN_SAMPLES = 50000 # arguments.n_train_samples
    N_TEST_SAMPLES = 10000 # arguments.n_test_samples #! used to be 10000

else:
    N_TRAIN_SAMPLES = 5000
    N_TEST_SAMPLES = 1000
    TRAIN_BATCH_SIZE = arguments.batch_size # 20
    TEST_BATCH_SIZE = arguments.batch_size
N_TRAIN_BATCH = int(N_TRAIN_SAMPLES / TRAIN_BATCH_SIZE)
N_TEST_BATCH = int(N_TEST_SAMPLES / TEST_BATCH_SIZE)
TRAIN_PRINT_PERIOD = 0.1
TRAIN_PRINT_PERIOD_STEP = int(N_TRAIN_SAMPLES * TRAIN_PRINT_PERIOD / TRAIN_BATCH_SIZE)
TEST_PERIOD = 1.0  # Evaluate on test batch every TEST_PERIOD epochs
TEST_PERIOD_STEP = int(N_TRAIN_SAMPLES * TEST_PERIOD / TRAIN_BATCH_SIZE)
LEARNING_RATE = arguments.learning_rate
LR_DECAY_EPOCH = 5  # Perform decay very n epochs
LR_DECAY_FACTOR = 0.25
MIN_LEARNING_RATE = 1e-6
TARGET_FALSE = 5
TARGET_TRUE = 20

# Plot parameters
EXPORT_METRICS = True
EXPORT_DIR = Path("./output_metrics")
SAVE_DIR = Path("/cifar_"+str(CLUSTER)+"/"+str(N_HIDDEN_LAYERS)+"_"+ str(str(conv_var['filter']))+"_"+str(output_var['n_neurons'])+"_"+str(str(conv_var['filter'])))


def weight_initializer_conv(c: int, x: int, y: int, pre_c: int) -> cp.ndarray:
    return cp.random.uniform(-2.0, 2.0, size=(c, x, y, pre_c), dtype=cp.float32)


def weight_initializer_ff(n_post: int, n_pre: int) -> cp.ndarray:
    return cp.random.uniform(-1.0, 1.0, size=(n_post, n_pre), dtype=cp.float32)
    # return cp.random.uniform(1.0, 1.0, size=(n_post, n_pre), dtype=cp.float32)


for run in range(NUMBER_OF_RUNS):

    max_int = np.iinfo(np.int32).max
    # np_seed = 319596201

    if not FIX_SEED:
        np_seed = int(cp.random.randint(low=0, high=max_int))
    else:
        np_seed = 733255843
        print('fixing seed np', np_seed)
    if not FIX_SEED:
        cp_seed = int(cp.random.randint(low=0, high=max_int))
    else:
        cp_seed = 1598833012
        print('fixing seed cp', cp_seed)

    np.random.seed(np_seed)
    cp.random.seed(cp_seed)
    print(f"Numpy seed: {np_seed}, Cupy seed: {cp_seed}")

    if EXPORT_METRICS and not EXPORT_DIR.exists():
        EXPORT_DIR.mkdir()

    print("Loading datasets...")
    dataset = Dataset(target_dir=DATASET_PATH, use_multi_channel=USE_3_CHANNELS, cifar100=USE_CIFAR100, use_coarse_labels=USE_COURSE_LABELS)


    print("Creating network...")
    network = Network()
    build_network_SCNN(network, weight_initializer_conv, weight_initializer_ff, INPUT_SHAPE, STANDARD, N_HIDDEN_LAYERS, conv_var, conv_res_var, fc_var, output_var, USE_RESIDUAL, RESIDUAL_EVERY_N, RESIDUAL_JUMP_LENGTH, USE_PADDING, USE_DELAY)

    if TTFS:
        loss_fct = TTFSSoftmaxCrossEntropy(tau=0.005)
    else:
        loss_fct = SpikeCountClassLoss(target_false=TARGET_FALSE, target_true=TARGET_TRUE)
    optimizer = AdamOptimizer(learning_rate=LEARNING_RATE)
    for layer in network.layers:
        if layer._is_residual:
            print(layer.name, layer.jump_layer.name)
            print(layer.n_neurons)
        else:
            print(layer.name)
            print(layer.n_neurons)
    # Metrics
    training_steps = 0
    train_loss_monitor = LossMonitor(export_path=EXPORT_DIR / "loss_train")
    train_accuracy_monitor = AccuracyMonitor(export_path=EXPORT_DIR / "accuracy_train")
    train_silent_label_monitor = SilentLabelsMonitor()
    train_time_monitor = TimeMonitor()
    train_monitors_manager = MonitorsManager([train_loss_monitor,
                                              train_accuracy_monitor,
                                              train_silent_label_monitor,
                                              train_time_monitor],
                                             print_prefix="Train | ")

    test_loss_monitor = LossMonitor(export_path=EXPORT_DIR / "loss_test")
    test_accuracy_monitor = AccuracyMonitor(export_path=EXPORT_DIR / "accuracy_test")
    test_learning_rate_monitor = ValueMonitor(name="Learning rate", decimal=5)
    # Only monitor LIF layers
    test_spike_counts_monitors = {l: SpikeCountMonitor(l.name) for l in network.layers
                                  if (isinstance(l, LIFLayer) or isinstance(l, ConvLIFLayer))}
    test_silent_monitors = {l: SilentNeuronsMonitor(l.name) for l in network.layers
                            if (isinstance(l, LIFLayer) or isinstance(l, ConvLIFLayer))}
    test_time_monitor = TimeMonitor()
    all_test_monitors = [test_loss_monitor, test_accuracy_monitor, test_learning_rate_monitor]
    all_test_monitors.extend(test_spike_counts_monitors.values())
    all_test_monitors.extend(test_silent_monitors.values())
    all_test_monitors.append(test_time_monitor)
    test_monitors_manager = MonitorsManager(all_test_monitors,
                                            print_prefix="Test | ")
    if USE_WANDB:
        w_b = wandb_handler("Final_thesis_testing", "CIFAR_conv_run_"+str(run),
        {
        "Cluster": CLUSTER,
        "Use_residual": USE_RESIDUAL,
        "loss": "SpikeCountClassLoss" if not TTFS else "TTFSSoftmaxCrossEntropy",
        "Use_delay": USE_DELAY,
        "Standard": STANDARD,
        "N_HIDDEN_LAYERS": N_HIDDEN_LAYERS,
        "batch_size": TRAIN_BATCH_SIZE,
        "residual_every_n": RESIDUAL_EVERY_N,
        "residual_jump_length": RESIDUAL_JUMP_LENGTH,
        "Use_residual": USE_RESIDUAL,
        "loss": "SpikeCountClassLoss" if not TTFS else "TTFSSoftmaxCrossEntropy",
        "use_padding": USE_PADDING,
        "n_of_train_samples": N_TRAIN_SAMPLES,
        "n_of_test_samples": N_TEST_SAMPLES,
        "channels": str(CHANNELS),
        "conv": str(conv_var),
        "conv_res": str(conv_res_var),
        "learning_rate": LEARNING_RATE,
        "architecture": "CNN",
        "dataset": "CIFAR100" if USE_CIFAR100 else "CIFAR10",
        "epochs": N_TRAINING_EPOCHS,
        "True_target": TARGET_TRUE,
        "False_target": TARGET_FALSE,
        },
        True)
    else:
        w_b = None

    best_acc = 0.0
    tracker = [0.0]* len(network.layers)
    print("Training...")
    dic = Path("last"+ str(SAVE_DIR))
    if RESTORE and dic.exists():
        print("""Restoring network from last save...""", dic)
        network.restore(dic)
    for epoch in range(N_TRAINING_EPOCHS):
        train_time_monitor.start()
        if not FIX_SEED:
            dataset.shuffle()
        # ! remove the shuffle for testability

        # Learning rate decay
        if epoch > 0 and epoch % LR_DECAY_EPOCH == 0:
            optimizer.learning_rate = np.maximum(LR_DECAY_FACTOR * optimizer.learning_rate, MIN_LEARNING_RATE)

        for batch_idx in range(N_TRAIN_BATCH):
            # Get next batch
            spikes, n_spikes, labels = dataset.get_train_batch(batch_idx, TRAIN_BATCH_SIZE, augment=True)

            # Inference
            network.reset()
            network.forward(spikes, n_spikes, max_simulation=SIMULATION_TIME, training=True)
            out_spikes, n_out_spikes = network.output_spike_trains

            # if (n_out_spikes != 0).any():
            #     print(f"Spikes")
            # else:
            #     print(f"No spikes")

            # Predictions, loss and errors
            pred = loss_fct.predict(out_spikes, n_out_spikes)
            loss, errors = loss_fct.compute_loss_and_errors(out_spikes, n_out_spikes, labels)

            pred_cpu = pred.get()
            loss_cpu = loss.get()
            n_out_spikes_cpu = n_out_spikes.get()

            # Update monitors
            train_loss_monitor.add(loss_cpu)
            train_accuracy_monitor.add(pred_cpu, labels)
            train_silent_label_monitor.add(n_out_spikes_cpu, labels)

            # Compute gradient
            gradient = network.backward(errors)
            # avg_gradient = [None if g is None else cp.mean(g, axis=0) for g, layer in zip(gradient, network.layers)]
            avg_gradient = []

            for g, layer in zip(gradient, network.layers):
                if g is None:
                    avg_gradient.append(None)
                elif layer._is_residual and isinstance(g, tuple):
                    grad_entry = []
                    for i in range(len(g)):
                        averaged_values = cp.mean(g[i], axis=0)
                        grad_entry.append(averaged_values)
                    avg_gradient.append(grad_entry)
                else:
                    averaged_values = cp.mean(g, axis=0)
                    avg_gradient.append(averaged_values)
            # for i in range(len(avg_gradient)):
            #     if i == 3:
            #         print("Gradient_avg: ", cp.max(avg_gradient[i][1]), cp.min(avg_gradient[i][1]), cp.mean(avg_gradient[i][1]))
            #         print("Gradient: ", cp.max(gradient[i][1]), cp.min(gradient[i][1]), cp.mean(gradient[i][1]))
            del gradient

            if FULL_METRIC:
                for i in range(len(avg_gradient)):
                    if avg_gradient[i] is not None:
                        if isinstance(avg_gradient[i], list):
                            for j in range(len(avg_gradient[i])):
                                tracker[i] = (tracker[i] + float(cp.mean(avg_gradient[i][j])))/2
                            if training_steps % TRAIN_PRINT_PERIOD_STEP == 0:
                                if USE_WANDB:
                                    w_b.save({"Mean Gradient Magnitude at residual layer "+str(i): tracker[i]})
                                print("Mean Gradient Magnitude at residual layer "+str(i)+": ", tracker[i])
                                tracker = [0.0]* len(network.layers)
                        else:
                            tracker[i] = (tracker[i] + float(cp.mean(avg_gradient[i])))/2
                            if training_steps % TRAIN_PRINT_PERIOD_STEP == 0:
                                if USE_WANDB:
                                    w_b.save({"Mean Gradient Magnitude at layer "+str(i): tracker[i]})
                                print("Mean Gradient Magnitude at layer "+str(i)+": ", tracker[i])
                                tracker = [0.0]* len(network.layers)
            # Apply step
            deltas = optimizer.step(avg_gradient)
            # for i in range(len(deltas)):
            #     if i == 3:
            #         print("Deltas: ", cp.max(deltas[i][1]), cp.min(deltas[i][1]), cp.mean(deltas[i][1]))
            del avg_gradient

            network.apply_deltas(deltas)
            del deltas

            training_steps += 1
            # print(f"Training steps: {training_steps}")
            epoch_metrics = training_steps * TRAIN_BATCH_SIZE / N_TRAIN_SAMPLES

            # Training metrics
            if training_steps % TRAIN_PRINT_PERIOD_STEP == 0:
                # Compute metrics

                train_monitors_manager.record(epoch_metrics)
                train_monitors_manager.print(epoch_metrics, use_wandb=USE_WANDB, w_b = w_b)
                train_monitors_manager.export()
                out_copy = cp.copy(out_spikes)
                mask = cp.isinf(out_copy)
                out_copy[mask] = cp.nan
                mean_spikes_for_times = cp.nanmean(out_copy)

                first_spike_for_times = cp.nanmin(out_copy)

                mean_res = cp.mean(cp.array(mean_spikes_for_times))
                mean_first = cp.mean(cp.array(first_spike_for_times))
                if FULL_METRIC:
                    print(f'Output layer mean times: {mean_res}')
                    print(f'Output layer first spike: {mean_first}')
                if USE_WANDB:
                    w_b.save({"Train_mean_spikes_for_times": float(mean_res), "Train_first_spike_for_times": float(mean_first)})
                

            # Test evaluation
            if training_steps % TEST_PERIOD_STEP == 0:
                test_time_monitor.start()
                mean_spikes_for_times = []
                first_spike_for_times = []
                for batch_idx in range(N_TEST_BATCH):
                    spikes, n_spikes, labels = dataset.get_test_batch(batch_idx, TEST_BATCH_SIZE)
                    network.reset()
                    network.forward(spikes, n_spikes, max_simulation=SIMULATION_TIME)
                    out_spikes, n_out_spikes = network.output_spike_trains

                    out_copy = cp.copy(out_spikes)
                    mask = cp.isinf(out_copy)
                    out_copy[mask] = cp.nan
                    mean_spikes_for_times.append(cp.nanmean(out_copy))

                    first_spike_for_times.append(cp.nanmin(out_copy))

                    pred = loss_fct.predict(out_spikes, n_out_spikes)
                    loss = loss_fct.compute_loss(out_spikes, n_out_spikes, labels)

                    pred_cpu = pred.get()
                    loss_cpu = loss.get()
                    test_loss_monitor.add(loss_cpu)
                    test_accuracy_monitor.add(pred_cpu, labels)

                    for l, mon in test_spike_counts_monitors.items():
                        mon.add(l.spike_trains[1])

                    for l, mon in test_silent_monitors.items():
                        mon.add(l.spike_trains[1])

                test_learning_rate_monitor.add(optimizer.learning_rate)

                records = test_monitors_manager.record(epoch_metrics)
                test_monitors_manager.print(epoch_metrics, use_wandb=USE_WANDB, w_b = w_b)
                test_monitors_manager.export()
                
                mean_res = cp.mean(cp.array(mean_spikes_for_times))
                mean_first = cp.mean(cp.array(first_spike_for_times))
                if not CLUSTER:
                    print(f'Output layer mean times: {mean_res}')
                    print(f'Output layer first spike: {mean_first}')
                if USE_WANDB:
                    w_b.save({"Test_mean_spikes_for_times": float(mean_res), "Test_first_spike_for_times": float(mean_first)})


                acc = records[test_accuracy_monitor]
                dic = Path("last" + str(SAVE_DIR))
                network.store(dic)
                if acc > best_acc:
                    best_acc = acc
                    dic = Path("best" + str(SAVE_DIR))
                    network.store(dic)
                    print(f"Best accuracy: {np.around(best_acc, 2)}%, Networks NOT save to: {SAVE_DIR}")
        if USE_WANDB:
            w_b.log()
    if USE_WANDB:
        w_b.finish()
    print("Done!: ", run)   