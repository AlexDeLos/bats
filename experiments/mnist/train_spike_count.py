from pathlib import Path
import cupy as cp
import numpy as np
import os
import wandb
import random
import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# sys.path.insert(0, "../../")  # Add repository root to python path

from Dataset import Dataset
from bats.Monitors import *
from bats.Layers import InputLayer, LIFLayer, LIFLayerResidual
from bats.Losses import *
from bats.Network import Network
from bats.Optimizers import *

# Dataset
# DATASET_PATH = Path("../../datasets/mnist.npz")
DATASET_PATH = Path("./datasets/mnist.npz")

N_INPUTS = 28 * 28
SIMULATION_TIME = 0.2

# Hidden layer
N_NEURONS_1 = 42 #!800 #? Should I lower it?
TAU_S_1 = 0.130
THRESHOLD_HAT_1 = 0.2
DELTA_THRESHOLD_1 = 1 * THRESHOLD_HAT_1
SPIKE_BUFFER_SIZE_1 = 30

# Output_layer
N_OUTPUTS = 10
TAU_S_OUTPUT = 0.130
THRESHOLD_HAT_OUTPUT = 1.3
DELTA_THRESHOLD_OUTPUT = 1 * THRESHOLD_HAT_OUTPUT
SPIKE_BUFFER_SIZE_OUTPUT = 30

#Residual parameters
USE_RESIDUAL = True
RESIDUAL_EVERY_N = -1
N_HIDDEN_LAYERS = 2
#!PROBLEM: when hidden layer > 1 and residual is used
# Training parameters
N_TRAINING_EPOCHS = 10 #! used to  be 100
N_TRAIN_SAMPLES = 6000 #! used to be 60000
N_TEST_SAMPLES = 1000 #! used to be 10000	
TRAIN_BATCH_SIZE = 50 #! used to be 50
TEST_BATCH_SIZE = 100
N_TRAIN_BATCH = int(N_TRAIN_SAMPLES / TRAIN_BATCH_SIZE)
N_TEST_BATCH = int(N_TEST_SAMPLES / TEST_BATCH_SIZE)
TRAIN_PRINT_PERIOD = 0.1
TRAIN_PRINT_PERIOD_STEP = int(N_TRAIN_SAMPLES * TRAIN_PRINT_PERIOD / TRAIN_BATCH_SIZE)
TEST_PERIOD = 1.0  # Evaluate on test batch every TEST_PERIOD epochs
TEST_PERIOD_STEP = int(N_TRAIN_SAMPLES * TEST_PERIOD / TRAIN_BATCH_SIZE)
LEARNING_RATE = 0.003
LR_DECAY_EPOCH = 10  # Perform decay very n epochs
LR_DECAY_FACTOR = 1.0
MIN_LEARNING_RATE = 0
TARGET_FALSE = 3
TARGET_TRUE = 15

best_acc_array = []

# Plot parameters
EXPORT_METRICS = True
EXPORT_DIR = Path("./experiments/mnist/output_metrics_REAL"+"-" + str(USE_RESIDUAL)+"-" +str(N_HIDDEN_LAYERS)+"-"+" hidden every " +str(RESIDUAL_EVERY_N) +"th Version 2")
SAVE_DIR = Path("./experiments/mnist/best_model")




def weight_initializer(n_post: int, n_pre: int) -> cp.ndarray:
    return cp.random.uniform(-1.0, 1.0, size=(n_post, n_pre), dtype=cp.float32)


if __name__ == "__main__":
    

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


    
    # uilding the network
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
            
        elif i == N_HIDDEN_LAYERS - 1 and USE_RESIDUAL:
            #TODO: FIX PROBLEM WITH: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered Exception ignored in: 'cupy.cuda.function.Module.__dealloc__'
            hidden_layer = LIFLayerResidual(previous_layer=hidden_layers[i-1], jump_layer= input_layer, n_neurons=N_NEURONS_1, tau_s=TAU_S_1,
                                    theta=THRESHOLD_HAT_1,
                                    delta_theta=DELTA_THRESHOLD_1,
                                    weight_initializer=weight_initializer,
                                    max_n_spike=SPIKE_BUFFER_SIZE_1,
                                    name="Residual layer " + str(i))
        # elif i % RESIDUAL_EVERY_N ==0 and USE_RESIDUAL:
        #     hidden_layer = LIFLayerResidual(previous_layer=hidden_layers[i-1], jump_layer= hidden_layers[i - RESIDUAL_EVERY_N], n_neurons=N_NEURONS_1, tau_s=TAU_S_1,
        #                             theta=THRESHOLD_HAT_1,
        #                             delta_theta=DELTA_THRESHOLD_1,
        #                             weight_initializer=weight_initializer,
        #                             max_n_spike=SPIKE_BUFFER_SIZE_1,
        #                             name="Residual layer " + str(i))


        # elif (i == N_HIDDEN_LAYERS - 1 or i % RESIDUAL_EVERY_N ==0) and N_HIDDEN_LAYERS > 5 and USE_RESIDUAL:
        #     hidden_layer = LIFLayerResidual(previous_layer=hidden_layers[i-1], jump_layer= input_layer, n_neurons=N_NEURONS_1, tau_s=TAU_S_1,
        #                             theta=THRESHOLD_HAT_1,
        #                             delta_theta=DELTA_THRESHOLD_1,
        #                             weight_initializer=weight_initializer,
        #                             max_n_spike=SPIKE_BUFFER_SIZE_1,
        #                             name="Residual layer " + str(i))
        else:
            hidden_layer = LIFLayer(previous_layer=hidden_layers[i-1], n_neurons=N_NEURONS_1, tau_s=TAU_S_1,
                                    theta=THRESHOLD_HAT_1,
                                    delta_theta=DELTA_THRESHOLD_1,
                                    weight_initializer=weight_initializer,
                                    max_n_spike=SPIKE_BUFFER_SIZE_1,
                                    name="Hidden layer " + str(i))
        hidden_layers.append(hidden_layer)
        network.add_layer(hidden_layer)

    output_layer = LIFLayer(previous_layer=hidden_layer, n_neurons=N_OUTPUTS, tau_s=TAU_S_OUTPUT,
                            theta=THRESHOLD_HAT_OUTPUT,
                            delta_theta=DELTA_THRESHOLD_OUTPUT,
                            weight_initializer=weight_initializer,
                            max_n_spike=SPIKE_BUFFER_SIZE_OUTPUT,
                            name="Output layer")
    network.add_layer(output_layer)
    #End of network building

    loss_fct = SpikeCountClassLoss(target_false=TARGET_FALSE, target_true=TARGET_TRUE)
    optimizer = AdamOptimizer(learning_rate=LEARNING_RATE)

    # Metrics
        #Weights and biases
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="Residual-SNN",
        
        # track hyperparameters and run metadata4
        config={
        "N_HIDDEN_LAYERS": N_HIDDEN_LAYERS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "residual_every_n": RESIDUAL_EVERY_N,
        "use_residual": USE_RESIDUAL,
        "n_of_train_samples": N_TRAIN_SAMPLES,
        "n_of_test_samples": N_TEST_SAMPLES,
        "n_neurons": N_NEURONS_1,
        "learning_rate": LEARNING_RATE,
        "architecture": "SNN",
        "dataset": "MNIST",
        "epochs": N_TRAINING_EPOCHS,
        "version": "test",
        }
    )
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
    test_spike_counts_monitors = {l: SpikeCountMonitor(l.name) for l in network.layers if isinstance(l, LIFLayer) or isinstance(l, LIFLayerResidual)}
    test_silent_monitors = {l: SilentNeuronsMonitor(l.name) for l in network.layers if isinstance(l, LIFLayer) or isinstance(l, LIFLayerResidual)}
    test_norm_monitors = {l: WeightsNormMonitor(l.name, export_path=EXPORT_DIR / ("weight_norm_" + l.name))
                        for l in network.layers if isinstance(l, LIFLayer) or isinstance(l, LIFLayerResidual)}
    test_time_monitor = TimeMonitor()
    all_test_monitors = [test_loss_monitor, test_accuracy_monitor, test_learning_rate_monitor]
    all_test_monitors.extend(test_spike_counts_monitors.values())
    all_test_monitors.extend(test_silent_monitors.values())
    all_test_monitors.extend(test_norm_monitors.values())
    all_test_monitors.append(test_time_monitor)
    test_monitors_manager = MonitorsManager(all_test_monitors,
                                            print_prefix="Test | ")

    best_acc = 0.0
    print("Training...")
    for epoch in range(N_TRAINING_EPOCHS):
        train_time_monitor.start()
        dataset.shuffle()

        # Learning rate decay
        if epoch > 0 and epoch % LR_DECAY_EPOCH == 0:
            optimizer.learning_rate = np.maximum(LR_DECAY_FACTOR * optimizer.learning_rate, MIN_LEARNING_RATE)

        for batch_idx in range(N_TRAIN_BATCH):
            # print("batch_idx: ", batch_idx)
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
            avg_gradient = [None if g is None else cp.mean(g, axis=0) for g, layer in zip(gradient, network.layers)]
            del gradient

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
                train_monitors_manager.print(epoch_metrics)
                train_monitors_manager.export()
                out_spikes, n_out_spikes = network.output_spike_trains
                mask = cp.isinf(out_spikes)
                out_spikes[mask] = cp.nan
                mean_spikes_for_times = cp.nanmean(out_spikes)
                first_spike_for_times = cp.nanmin(out_spikes)
                print(f'Output layer mean times: {mean_spikes_for_times}')
                print(f'Output layer first spike: {first_spike_for_times}')
                # with open('times.txt', 'a') as f:
                #     string = f'Train Step Number: {training_steps/TRAIN_PRINT_PERIOD_STEP}' + "\n"+ f'Output layer mean times: {mean_spikes_for_times}' + "\n" + f'Output layer first spike: {first_spike_for_times}' + "\n" + "-------------------------------------"+"\n"
                #     f.write(string)

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
                test_monitors_manager.print(epoch_metrics)
                test_monitors_manager.export()

                acc = records[test_accuracy_monitor]
                loss_to_save = records[test_loss_monitor]
                wandb.log({"acc": acc, "loss": loss_to_save, "time_to_output": mean_spikes_for_times.item(), "first_spike": first_spike_for_times.item()})

                if acc > best_acc:
                    best_acc = acc
                    network.store(SAVE_DIR)
                    # print(f"Best accuracy: {np.around(best_acc, 2)}%, Networks save to: {SAVE_DIR}")
    best_acc_array.append(best_acc)    
        
    wandb.finish()

