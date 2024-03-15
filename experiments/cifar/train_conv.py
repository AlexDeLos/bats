from pathlib import Path
import wandb
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

from bats.Layers.ConvLIFLayerResidual_2 import ConvLIFLayerResidual_2
from bats.Utils.utils import get_arguments

from bats.Layers.PoolingLayer import PoolingLayer

arguments = get_arguments()
# Residual arguments
N_HIDDEN_LAYERS = arguments.n_hidden_layers
USE_RESIDUAL = arguments.use_residual
RESIDUAL_EVERY_N = arguments.residual_every_n

# Change from small test on computer to big test on cluster
CLUSTER = arguments.cluster
USE_WANDB = arguments.use_wanb
ALTERNATE = arguments.alternate
USE_RESIDUAL = arguments.use_residual
FIX_SEED = False
USE_PADDING = True #! residual and padd gives nans
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

if CLUSTER:
    FILTER_1 = np.array([5, 5, 12]) #? could it be the size of this filter's channels?
else:
    FILTER_1 = np.array([5, 5, 12])
TAU_S_1 = 0.130
THRESHOLD_HAT_1 = 0.04
DELTA_THRESHOLD_1 = 1 * THRESHOLD_HAT_1
SPIKE_BUFFER_SIZE_1 = 10

N_NEURONS_FC = 100
TAU_S_FC = 0.130
THRESHOLD_HAT_FC = 0.06
DELTA_THRESHOLD_FC = 1 * THRESHOLD_HAT_FC
SPIKE_BUFFER_SIZE_FC = 10

# Output_layer
if USE_COURSE_LABELS and USE_CIFAR100:
    N_OUTPUTS = 20
elif USE_CIFAR100:
    N_OUTPUTS = 100
else:
    N_OUTPUTS = 10
TAU_S_OUTPUT = 0.130
THRESHOLD_HAT_OUTPUT = 0.3
DELTA_THRESHOLD_OUTPUT = 1 * THRESHOLD_HAT_OUTPUT
SPIKE_BUFFER_SIZE_OUTPUT = 30

# Training parameters
N_TRAINING_EPOCHS = arguments.n_epochs #! used to  be 100
if CLUSTER:
    N_TRAIN_SAMPLES = arguments.n_train_samples
    N_TEST_SAMPLES = arguments.n_test_samples #! used to be 10000
    TRAIN_BATCH_SIZE = arguments.batch_size #! used to be 50 -> putting it at 50 crashes the cluster when using append
    TEST_BATCH_SIZE = arguments.batch_size_test
else:
    N_TRAIN_SAMPLES = 5000
    N_TEST_SAMPLES = 1000
    TRAIN_BATCH_SIZE = 20
    TEST_BATCH_SIZE = 40
    TRAIN_BATCH_SIZE = 20 # 20
    TEST_BATCH_SIZE = 50
N_TRAIN_BATCH = int(N_TRAIN_SAMPLES / TRAIN_BATCH_SIZE)
N_TEST_BATCH = int(N_TEST_SAMPLES / TEST_BATCH_SIZE)
TRAIN_PRINT_PERIOD = 0.1
TRAIN_PRINT_PERIOD_STEP = int(N_TRAIN_SAMPLES * TRAIN_PRINT_PERIOD / TRAIN_BATCH_SIZE)
TEST_PERIOD = 1.0  # Evaluate on test batch every TEST_PERIOD epochs
TEST_PERIOD_STEP = int(N_TRAIN_SAMPLES * TEST_PERIOD / TRAIN_BATCH_SIZE)
LEARNING_RATE = arguments.learning_rate
LR_DECAY_EPOCH = 10  # Perform decay very n epochs
LR_DECAY_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-5
TARGET_FALSE = 3
TARGET_TRUE = 30

# Plot parameters
EXPORT_METRICS = True
EXPORT_DIR = Path("./output_metrics")
SAVE_DIR = Path("./best_model")


def weight_initializer_conv(c: int, x: int, y: int, pre_c: int) -> cp.ndarray:
    return cp.random.uniform(-1.0, 1.0, size=(c, x, y, pre_c), dtype=cp.float32)


def weight_initializer_ff(n_post: int, n_pre: int) -> cp.ndarray:
    return cp.random.uniform(-1.0, 1.0, size=(n_post, n_pre), dtype=cp.float32)
    # return cp.random.uniform(1.0, 1.0, size=(n_post, n_pre), dtype=cp.float32)


for run in range(NUMBER_OF_RUNS):
    if USE_WANDB:
        wandb.init(
        # set the wandb project where this run will be logged
        project="Final_thesis_testing",
        name="CNN_"+ str(USE_RESIDUAL)+"# hidden_"+ str(N_HIDDEN_LAYERS) +" using CIFAR100? = "+str(USE_CIFAR100),
        
        # track hyperparameters and run metadata4
        config={
        "Cluster": CLUSTER,
        "Use_residual": USE_RESIDUAL,
        "N_HIDDEN_LAYERS": N_HIDDEN_LAYERS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "residual_every_n": RESIDUAL_EVERY_N,
        "use_padding": USE_PADDING,
        "n_of_train_samples": N_TRAIN_SAMPLES,
        "n_of_test_samples": N_TEST_SAMPLES,
        "Filter": str(FILTER_1),
        "learning_rate": LEARNING_RATE,
        "architecture": "CNN",
        "dataset": "CIFAR-100" if USE_CIFAR100 else "CIFAR-10",
        "epochs": N_TRAINING_EPOCHS,
        "version": "1.0.0_cluster_" + str(CLUSTER),
        }
        )



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
    input_layer = ConvInputLayer(neurons_shape=INPUT_SHAPE, name="Input layer")
    network.add_layer(input_layer, input=True)

    #! Standard network builder
    # hidden_layers = []
    # for i in range(N_HIDDEN_LAYERS):
    #     if i == 0:
    #         conv = ConvLIFLayer(previous_layer=input_layer, filters_shape=FILTER_1, use_padding=USE_PADDING,
    #                         tau_s=TAU_S_1,
    #                         filter_from_next=FILTER_1,
    #                         theta=THRESHOLD_HAT_1,
    #                         delta_theta=DELTA_THRESHOLD_1,
    #                         weight_initializer=weight_initializer_conv,
    #                         max_n_spike=SPIKE_BUFFER_SIZE_1,
    #                         name="Convolution "+str(i))
    #     elif i % RESIDUAL_EVERY_N == 0 or i == N_HIDDEN_LAYERS-1:
    #         if USE_RESIDUAL and i == N_HIDDEN_LAYERS-1:
    #             conv = ConvLIFLayerResidual_2(previous_layer=network.layers[-1], jump_layer=hidden_layers[i-RESIDUAL_EVERY_N], filters_shape=FILTER_1, use_padding=USE_PADDING,
    #                         tau_s=TAU_S_1,
    #                         theta=THRESHOLD_HAT_1,
    #                         delta_theta=DELTA_THRESHOLD_1,
    #                         weight_initializer=weight_initializer_conv,
    #                         max_n_spike=SPIKE_BUFFER_SIZE_1,
    #                         name="Convolution Residual "+str(i))
    #         elif USE_RESIDUAL:
    #             conv = ConvLIFLayerResidual_2(previous_layer=network.layers[-1], jump_layer=hidden_layers[i-RESIDUAL_EVERY_N], filters_shape=FILTER_1, use_padding=USE_PADDING,
    #                         tau_s=TAU_S_1,
    #                         filter_from_next=FILTER_1,
    #                         theta=THRESHOLD_HAT_1,
    #                         delta_theta=DELTA_THRESHOLD_1,
    #                         weight_initializer=weight_initializer_conv,
    #                         max_n_spike=SPIKE_BUFFER_SIZE_1,
    #                         name="Convolution Residual "+str(i))
    #         else:
    #             conv = ConvLIFLayer(previous_layer=network.layers[-1], filters_shape=FILTER_1, use_padding=USE_PADDING,
    #                         tau_s=TAU_S_1,
    #                         filter_from_next=FILTER_1,
    #                         theta=THRESHOLD_HAT_1,
    #                         delta_theta=DELTA_THRESHOLD_1,
    #                         weight_initializer=weight_initializer_conv,
    #                         max_n_spike=SPIKE_BUFFER_SIZE_1,
    #                         name="Convolution "+str(i))
    #     else:
    #         conv = ConvLIFLayer(previous_layer=conv, filters_shape=FILTER_1, use_padding=USE_PADDING,
    #                         tau_s=TAU_S_1,
    #                         filter_from_next=FILTER_1,
    #                         theta=THRESHOLD_HAT_1,
    #                         delta_theta=DELTA_THRESHOLD_1,
    #                         weight_initializer=weight_initializer_conv,
    #                         max_n_spike=SPIKE_BUFFER_SIZE_1,
    #                         name="Convolution "+str(i))
    #     hidden_layers.append(conv)
    #     network.add_layer(conv)
    #! end of standard network builder

    # pool_2 = PoolingLayer(conv, name="Pooling 2")
    # network.add_layer(pool_2)
    conv_1 = ConvLIFLayer(previous_layer=input_layer, filters_shape=np.array([5, 5, 32]), use_padding=USE_PADDING,
                        #   filter_from_next = FILTER_FROM_NEXT,
                          tau_s=TAU_S_1,
                          theta=THRESHOLD_HAT_1,
                          delta_theta=DELTA_THRESHOLD_1,
                          weight_initializer=weight_initializer_conv,
                          max_n_spike=SPIKE_BUFFER_SIZE_1,
                          name="Convolution 1")
    network.add_layer(conv_1)

    pool_1 = PoolingLayer(conv_1, name="Pooling 1")
    network.add_layer(pool_1)

    conv_2 = ConvLIFLayer(previous_layer=pool_1, filters_shape=np.array([5, 5, 32]), use_padding=USE_PADDING,
                        #   filter_from_next = FILTER_FROM_NEXT,
                          tau_s=TAU_S_1,
                          theta=THRESHOLD_HAT_1,
                          delta_theta=DELTA_THRESHOLD_1,
                          weight_initializer=weight_initializer_conv,
                          max_n_spike=SPIKE_BUFFER_SIZE_1,
                          name="Convolution 2")
    network.add_layer(conv_2)

    pool_2 = PoolingLayer(conv_2, name="Pooling 2")
    network.add_layer(pool_2)

    conv_3 = ConvLIFLayer(previous_layer=pool_2, filters_shape=np.array([5, 5, 32]), use_padding=USE_PADDING,
                    #   filter_from_next = FILTER_FROM_NEXT,
                        tau_s=TAU_S_1,
                        theta=THRESHOLD_HAT_1,
                        delta_theta=DELTA_THRESHOLD_1,
                        weight_initializer=weight_initializer_conv,
                        max_n_spike=SPIKE_BUFFER_SIZE_1,
                        name="Convolution 3")
    network.add_layer(conv_3)

    pool_3 = PoolingLayer(conv_3, name="Pooling 3")
    network.add_layer(pool_3)

    feedforward = LIFLayer(previous_layer=pool_3, n_neurons=N_NEURONS_FC, tau_s=TAU_S_FC,
                           theta=THRESHOLD_HAT_FC,
                           delta_theta=DELTA_THRESHOLD_FC,
                           weight_initializer=weight_initializer_ff,
                           max_n_spike=SPIKE_BUFFER_SIZE_FC,
                           name="Feedforward 1")
    network.add_layer(feedforward)

    output_layer = LIFLayer(previous_layer=feedforward, n_neurons=N_OUTPUTS, tau_s=TAU_S_OUTPUT,
                            theta=THRESHOLD_HAT_OUTPUT,
                            delta_theta=DELTA_THRESHOLD_OUTPUT,
                            weight_initializer=weight_initializer_ff,
                            max_n_spike=SPIKE_BUFFER_SIZE_OUTPUT,
                            name="Output layer")
    network.add_layer(output_layer)

    loss_fct = SpikeCountClassLoss(target_false=TARGET_FALSE, target_true=TARGET_TRUE)
    optimizer = AdamOptimizer(learning_rate=LEARNING_RATE)

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

    best_acc = 0.0
    tracker = [0.0]* len(network.layers)
    print("Training...")
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
                elif layer._is_residual:
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

            if USE_WANDB:
                for i in range(len(avg_gradient)):
                    if avg_gradient[i] is not None:
                        if isinstance(avg_gradient[i], list):
                            for j in range(len(avg_gradient[i])):
                                tracker[i] = (tracker[i] + float(cp.mean(cp.abs(avg_gradient[i][j]))))/2
                            if training_steps % TRAIN_PRINT_PERIOD_STEP == 0:
                                wandb.log({"Mean Gradient Magnitude at residual layer "+str(i): tracker[i]})
                                if not CLUSTER:
                                    print("Mean Gradient Magnitude at residual layer "+str(i)+": ", tracker[i])
                                tracker = [0.0]* len(network.layers)
                        else:
                            tracker[i] = (tracker[i] + float(cp.mean(cp.abs(avg_gradient[i]))))/2
                            if training_steps % TRAIN_PRINT_PERIOD_STEP == 0:
                                wandb.log({"Mean Gradient Magnitude at layer "+str(i): tracker[i]})
                                if not CLUSTER:
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
            epoch_metrics = training_steps * TRAIN_BATCH_SIZE / N_TRAIN_SAMPLES

            # Training metrics
            if training_steps % TRAIN_PRINT_PERIOD_STEP == 0:
                # Compute metrics

                train_monitors_manager.record(epoch_metrics)
                train_monitors_manager.print(epoch_metrics, use_wandb=USE_WANDB)
                train_monitors_manager.export()
                out_copy = cp.copy(out_spikes)
                mask = cp.isinf(out_copy)
                out_copy[mask] = cp.nan
                mean_spikes_for_times = cp.nanmean(out_copy)

                first_spike_for_times = cp.nanmin(out_copy)

                mean_res = cp.mean(cp.array(mean_spikes_for_times))
                mean_first = cp.mean(cp.array(first_spike_for_times))
                if not CLUSTER:
                    print(f'Output layer mean times: {mean_res}')
                    print(f'Output layer first spike: {mean_first}')
                if USE_WANDB:
                    wandb.log({"Train_mean_spikes_for_times": float(mean_res), "Train_first_spike_for_times": float(mean_first)})
                

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
                test_monitors_manager.print(epoch_metrics, use_wandb=USE_WANDB)
                test_monitors_manager.export()
                
                mean_res = cp.mean(cp.array(mean_spikes_for_times))
                mean_first = cp.mean(cp.array(first_spike_for_times))
                if not CLUSTER:
                    print(f'Output layer mean times: {mean_res}')
                    print(f'Output layer first spike: {mean_first}')
                if USE_WANDB:
                    wandb.log({"Test_mean_spikes_for_times": float(mean_res), "Test_first_spike_for_times": float(mean_first)})


                acc = records[test_accuracy_monitor]
                if acc > best_acc:
                    best_acc = acc
                    # network.store(SAVE_DIR)
                    print(f"Best accuracy: {np.around(best_acc, 2)}%, Networks NOT save to: {SAVE_DIR}")
    if USE_WANDB:
        wandb.finish()
    print("Done!: ", run)   