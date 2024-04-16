from operator import ne
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
from experiments.utils.utils import build_network_SNN, wandb_handler

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
neuron_var = {
    'n_neurons': 800,
    'tau_s': 0.130,
    'threshold_hat': 0.2,
    'delta_threshold': 1 * 0.2,
    'spike_buffer_size': 5
}
neuron_out_var = {
    'n_neurons': 10,
    'tau_s': 0.130,
    'threshold_hat': 0.7,
    'delta_threshold': 1 * 0.7,
    'spike_buffer_size': 20
}
neuron_res_var = {
    'n_neurons': 800,
    'tau_s': 0.130,
    'threshold_hat': 0.25,
    'delta_threshold': 1 * 0.25,
    'spike_buffer_size': 5
}

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
    build_network_SNN(network, weight_initializer, N_INPUTS, N_HIDDEN_LAYERS, neuron_var, neuron_out_var, neuron_res_var, USE_RESIDUAL, RESIDUAL_EVERY_N, RESIDUAL_JUMP_LENGTH, FUSE_FUNCTION)
    
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
        w_b = wandb_handler("Final_thesis_testing", "fashion_MLP_run_"+str(run),
        {"Cluster": CLUSTER,
        "Use_residual": USE_RESIDUAL,
        "N_HIDDEN_LAYERS": N_HIDDEN_LAYERS,
        "residual_every_n": RESIDUAL_EVERY_N,
        "residual_jump_length": RESIDUAL_JUMP_LENGTH,
        "n_of_train_samples": N_TRAIN_SAMPLES,
        "n_of_test_samples": N_TEST_SAMPLES,
        "learning_rate": LEARNING_RATE,
        "batch_size": TRAIN_BATCH_SIZE,
        "dataset": "fashion MNIST",
        "epochs": N_TRAINING_EPOCHS,
        "Fuse_function": FUSE_FUNCTION,
        "neuron_var": str(neuron_var),
        "neuron_out_var": str(neuron_out_var),
        "neuron_res_var": str(neuron_res_var),},
        False)

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
            del avg_gradient

            network.apply_deltas(deltas)
            del deltas

            training_steps += 1
            epoch_metrics = training_steps * TRAIN_BATCH_SIZE / N_TRAIN_SAMPLES

            # Training metrics
            if training_steps % TRAIN_PRINT_PERIOD_STEP == 0:
                # Compute metrics

                train_monitors_manager.record(epoch_metrics)
                train_monitors_manager.print(epoch_metrics, use_wandb=USE_WANDB, w_b=w_b)
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
                test_monitors_manager.print(epoch_metrics, use_wandb=USE_WANDB, w_b= w_b)
                test_monitors_manager.export()
                
                mean_res = cp.mean(cp.array(mean_spikes_for_times))
                mean_first = cp.mean(cp.array(first_spike_for_times))
                if not CLUSTER:
                    print(f'Output layer mean times: {mean_res}')
                    print(f'Output layer first spike: {mean_first}')
                if USE_WANDB:
                    w_b.save({"Test_mean_spikes_for_times": float(mean_res), "Test_first_spike_for_times": float(mean_first)})

                acc = records[test_accuracy_monitor]
                if acc > best_acc:
                    best_acc = acc
                    # network.store(SAVE_DIR)
                    print(f"Best accuracy: {np.around(best_acc, 2)}%, Networks NOT save to: {SAVE_DIR}")
            if USE_WANDB and ((training_steps % TRAIN_PRINT_PERIOD_STEP == 0) or (training_steps % TEST_PERIOD_STEP == 0)):
                w_b.log()
    if USE_WANDB:
        w_b.finish()
    print("Done!: ", run)   
