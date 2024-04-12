from pathlib import Path
import cupy as cp
import numpy as np
import wandb

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Dataset import Dataset
from bats.Monitors import *
from bats.Layers import InputLayer, LIFLayer, LIFLayerResidual
from bats.Losses import *
from bats.Network import Network
from bats.Optimizers import *
from bats.Utils.utils import get_arguments

DATASET_PATH = Path("./datasets/")

arguments = get_arguments()
# Residual arguments
N_HIDDEN_LAYERS = arguments.n_hidden_layers
USE_RESIDUAL = arguments.use_residual
print("Using residual: ", USE_RESIDUAL)
RESIDUAL_EVERY_N = arguments.residual_every_n
FUSE_FUNCTION = arguments.fuse_func
LEARNING_RATE = arguments.learning_rate
FULL_METRIC = False

#Residual parameters
RESIDUAL_JUMP_LENGTH = arguments.residual_jump_length

CLUSTER = arguments.cluster
USE_WANDB = arguments.use_wanb
ALTERNATE = arguments.alternate


NUMBER_OF_RUNS = arguments.runs

N_INPUTS = 28 * 28
SIMULATION_TIME = 0.2

# Hidden layer
N_NEURONS_1 = 800
TAU_S_1 = 0.130
THRESHOLD_HAT_1 = 0.1
DELTA_THRESHOLD_1 = 1 * THRESHOLD_HAT_1
SPIKE_BUFFER_SIZE_1 = 30
# Residual layer
if CLUSTER:
    N_NEURONS_RES = 400 #!800 #? Should I lower it?
else:
    N_NEURONS_RES = 400
TAU_S_RES = 0.130
THRESHOLD_HAT_RES = 0.1
DELTA_THRESHOLD_RES = 1 * THRESHOLD_HAT_RES
SPIKE_BUFFER_SIZE_RES = 5

# Output_layer
N_OUTPUTS = 10
TAU_S_OUTPUT = 0.130
THRESHOLD_HAT_OUTPUT = 0.7
DELTA_THRESHOLD_OUTPUT = 1 * THRESHOLD_HAT_OUTPUT
SPIKE_BUFFER_SIZE_OUTPUT = 20
# Training parameters
N_TRAINING_EPOCHS = arguments.n_epochs #! used to  be 100
if CLUSTER:
    N_TRAIN_SAMPLES = 60000
    N_TEST_SAMPLES = 10000
    TRAIN_BATCH_SIZE = arguments.batch_size
    TEST_BATCH_SIZE = arguments.batch_size
else:
    N_TRAIN_SAMPLES = 6000
    N_TEST_SAMPLES = 1000
    TRAIN_BATCH_SIZE = 50
    TEST_BATCH_SIZE = 100
# Training parameters

N_TRAIN_BATCH = int(N_TRAIN_SAMPLES / TRAIN_BATCH_SIZE)
N_TEST_BATCH = int(N_TEST_SAMPLES / TEST_BATCH_SIZE)
TRAIN_PRINT_PERIOD = 0.1
TRAIN_PRINT_PERIOD_STEP = int(N_TRAIN_SAMPLES * TRAIN_PRINT_PERIOD / TRAIN_BATCH_SIZE)
TEST_PERIOD = 1.0  # Evaluate on test batch every TEST_PERIOD epochs
TEST_PERIOD_STEP = int(N_TRAIN_SAMPLES * TEST_PERIOD / TRAIN_BATCH_SIZE)
LR_DECAY_EPOCH = 10  # Perform decay very n epochs
LR_DECAY_FACTOR = 1.0
MIN_LEARNING_RATE = 1e-4
TARGET_FALSE = 3
TARGET_TRUE = 15

# Plot parameters
EXPORT_METRICS = False
EXPORT_DIR = Path("output_metrics")
SAVE_DIR = Path("best_model")


def weight_initializer(n_post: int, n_pre: int) -> cp.ndarray:
    return cp.random.uniform(-1.0, 1.0, size=(n_post, n_pre), dtype=cp.float32)


for run in range(NUMBER_OF_RUNS):
    print(USE_RESIDUAL, CLUSTER, FUSE_FUNCTION, N_HIDDEN_LAYERS, RESIDUAL_EVERY_N, run)

    max_int = np.iinfo(np.int32).max
    np_seed = np.random.randint(low=0, high=max_int)
    cp_seed = np.random.randint(low=0, high=max_int)
    np.random.seed(np_seed)
    cp.random.seed(cp_seed)
    print(f"Numpy seed: {np_seed}, Cupy seed: {cp_seed}")

    if EXPORT_METRICS and not EXPORT_DIR.exists():
        EXPORT_DIR.mkdir()

    # Dataset
    print("Loading datasets...")
    dataset = Dataset(path=DATASET_PATH)

    print("Creating network...")
    network = Network()
    input_layer = InputLayer(n_neurons=N_INPUTS, name="Input layer")
    network.add_layer(input_layer, input=True)

    hidden_layers = []
    for i in range(N_HIDDEN_LAYERS):
        if i == 0:
            hidden_layer = LIFLayer(previous_layer=input_layer, n_neurons=N_NEURONS_1, tau_s=TAU_S_1,
                                    theta=THRESHOLD_HAT_1,
                                    delta_theta=DELTA_THRESHOLD_1,
                                    weight_initializer=weight_initializer,
                                    max_n_spike=SPIKE_BUFFER_SIZE_1,
                                    name="Hidden layer 0")
        elif i % RESIDUAL_EVERY_N ==0 and USE_RESIDUAL:
            if i - RESIDUAL_JUMP_LENGTH < 0:
                jump_layer = input_layer
            else:
                jump_layer = hidden_layers[i - RESIDUAL_JUMP_LENGTH]
            hidden_layer = LIFLayerResidual(previous_layer=hidden_layers[i-1],
                                            # jump_layer= hidden_layers[i-1],
                                            jump_layer = jump_layer,
                                            n_neurons=N_NEURONS_RES, tau_s=TAU_S_RES,
                                            theta=THRESHOLD_HAT_RES,
                                            fuse_function=FUSE_FUNCTION,
                                            delta_theta=DELTA_THRESHOLD_RES,
                                            weight_initializer=weight_initializer,
                                            max_n_spike=SPIKE_BUFFER_SIZE_RES,
                                            name="Residual layer " + str(i))
        elif i % RESIDUAL_EVERY_N ==0 and not USE_RESIDUAL:
            hidden_layer = LIFLayer(previous_layer=hidden_layers[i-1], n_neurons=N_NEURONS_1, tau_s=TAU_S_1,
                                    theta=THRESHOLD_HAT_1,
                                    delta_theta=DELTA_THRESHOLD_1,
                                    weight_initializer=weight_initializer,
                                    max_n_spike=SPIKE_BUFFER_SIZE_1,
                                    name="Hidden layer " + str(i))

        else:
            hidden_layer = LIFLayer(previous_layer=hidden_layers[i-1], n_neurons=N_NEURONS_1, tau_s=TAU_S_1,
                                    theta=THRESHOLD_HAT_1,
                                    delta_theta=DELTA_THRESHOLD_1,
                                    weight_initializer=weight_initializer,
                                    max_n_spike=SPIKE_BUFFER_SIZE_1,
                                    name="Hidden layer " + str(i))
        hidden_layers.append(hidden_layer)
        network.add_layer(hidden_layer)

    output_layer = LIFLayer(previous_layer=hidden_layer, n_neurons=N_OUTPUTS, tau_s=TAU_S_OUTPUT, # type: ignore
                            theta=THRESHOLD_HAT_OUTPUT,
                            delta_theta=DELTA_THRESHOLD_OUTPUT,
                            weight_initializer=weight_initializer,
                            max_n_spike=SPIKE_BUFFER_SIZE_OUTPUT,
                            name="Output layer")
    network.add_layer(output_layer)
    
    for layer in network.layers:
        if isinstance(layer, LIFLayerResidual):
            print(layer.name, layer.jump_layer.name)
        else:
            print(layer.name)
        print(layer.n_neurons)
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
    test_spike_counts_monitors = {l: SpikeCountMonitor(l.name) for l in network.layers if isinstance(l, LIFLayer)}
    test_silent_monitors = {l: SilentNeuronsMonitor(l.name) for l in network.layers if isinstance(l, LIFLayer)}
    test_norm_monitors = {l: WeightsNormMonitor(l.name, export_path=EXPORT_DIR / ("weight_norm_" + l.name))
                          for l in network.layers if isinstance(l, LIFLayer)}
    test_time_monitor = TimeMonitor()
    all_test_monitors = [test_loss_monitor, test_accuracy_monitor, test_learning_rate_monitor]
    all_test_monitors.extend(test_spike_counts_monitors.values())
    all_test_monitors.extend(test_silent_monitors.values())
    all_test_monitors.extend(test_norm_monitors.values())
    all_test_monitors.append(test_time_monitor)
    test_monitors_manager = MonitorsManager(all_test_monitors,
                                            print_prefix="Test | ")

    if USE_WANDB:
        wandb.init(
        # set the wandb project where this run will be logged
        project="Final_thesis_testing",
        name="Fashion_mnist_MLP_run_"+str(run),
        
        # track hyperparameters and run metadata4
        config={
        "Cluster": CLUSTER,
        "Use_residual": USE_RESIDUAL,
        "N_HIDDEN_LAYERS": N_HIDDEN_LAYERS,
        "residual_every_n": RESIDUAL_EVERY_N,
        "residual_jump_length": RESIDUAL_JUMP_LENGTH,
        "n_of_train_samples": N_TRAIN_SAMPLES,
        "n_of_test_samples": N_TEST_SAMPLES,
        "learning_rate": LEARNING_RATE,
        "batch_size": TRAIN_BATCH_SIZE,
        "architecture": "MLP",
        "dataset": "Fashion MNIST",
        "epochs": N_TRAINING_EPOCHS,
        "version": "1.0.1_cluster_" + str(CLUSTER),
        "Fuse_function": FUSE_FUNCTION,
        }
        )

    best_acc = 0.0
    tracker = [0.0]* len(network.layers)
    print("Training...")
    for epoch in range(N_TRAINING_EPOCHS):
        train_time_monitor.start()
        dataset.shuffle()

        # Learning rate decay
        if epoch > 0 and epoch % LR_DECAY_EPOCH == 0:
            optimizer.learning_rate = np.maximum(LR_DECAY_FACTOR * optimizer.learning_rate, MIN_LEARNING_RATE)

        for batch_idx in range(N_TRAIN_BATCH):
            # Get next batch
            spikes, n_spikes, labels = dataset.get_train_batch(batch_idx, TRAIN_BATCH_SIZE)

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
            avg_gradient = []

            for g, layer in zip(gradient, network.layers):
                if g is None:
                    avg_gradient.append(None)
                else:
                    averaged_values = cp.mean(g, axis=0)
                    avg_gradient.append(averaged_values)
            del gradient

            if FULL_METRIC:
                for i in range(len(avg_gradient)):
                    if avg_gradient[i] is not None:
                        if isinstance(avg_gradient[i], list):
                            for j in range(len(avg_gradient[i])):
                                tracker[i] = (tracker[i] + float(cp.mean(avg_gradient[i][j])))/2
                            if training_steps % TRAIN_PRINT_PERIOD_STEP == 0:
                                if USE_WANDB:
                                    wandb.log({"Mean Gradient Magnitude at residual layer "+str(i): tracker[i]})
                                print("Mean Gradient Magnitude at residual layer "+str(i)+": ", tracker[i])
                                tracker = [0.0]* len(network.layers)
                        else:
                            tracker[i] = (tracker[i] + float(cp.mean(avg_gradient[i])))/2
                            if training_steps % TRAIN_PRINT_PERIOD_STEP == 0:
                                if USE_WANDB:
                                    wandb.log({"Mean Gradient Magnitude at layer "+str(i): tracker[i]})
                                print("Mean Gradient Magnitude at layer "+str(i)+": ", tracker[i])
                                tracker = [0.0]* len(network.layers)
            # Apply step
            deltas = optimizer.step(avg_gradient)
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
                if FULL_METRIC:
                    print(f'Output layer mean times: {mean_res}')
                    print(f'Output layer first spike: {mean_first}')
                if USE_WANDB:
                    wandb.log({"Train_mean_spikes_for_times": float(mean_res), "Train_first_spike_for_times": float(mean_first)})

            # Test evaluation
            if training_steps % TEST_PERIOD_STEP == 0:
                test_time_monitor.start()
                for batch_idx in range(N_TEST_BATCH):
                    spikes, n_spikes, labels = dataset.get_test_batch(batch_idx, TEST_BATCH_SIZE)
                    network.reset()
                    network.forward(spikes, n_spikes, max_simulation=SIMULATION_TIME)
                    out_spikes, n_out_spikes = network.output_spike_trains

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

                for l, mon in test_norm_monitors.items():
                    mon.add(l.weights)

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
