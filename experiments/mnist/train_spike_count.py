from pathlib import Path
import cupy as cp
import numpy as np
import os
import wandb
import sys  

# if it is not working try going back to the pip 3.9 interpreter


# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# sys.path.insert(0, "../../")  # Add repository root to python path

from Dataset import Dataset
from bats.Monitors import *
from bats.Layers import InputLayer, LIFLayer, LIFLayerResidual
from bats.Losses import *
from bats.Network import Network
from bats.Optimizers import *
from bats.Utils.utils import get_arguments

arguments = get_arguments()

# Dataset
# DATASET_PATH = Path("../../datasets/mnist.npz")
DATASET_PATH = Path("./datasets/mnist.npz")

N_INPUTS = 28 * 28
SIMULATION_TIME = 0.2

# Change from small test on computer to big test on cluster
CLUSTER = arguments.cluster
USE_WANDB = arguments.use_wanb
ALTERNATE = arguments.alternate
FUSE_FUNCTION = arguments.fuse_func
#TODO: try to get the non append function to run out of memory

#Residual parameters
USE_RESIDUAL = arguments.use_residual
RESIDUAL_EVERY_N = arguments.residual_every_n
RESIDUAL_JUMP_LENGTH = arguments.residual_jump_length
N_HIDDEN_LAYERS = arguments.n_hidden_layers

if CLUSTER:
    NUMBER_OF_RUNS = arguments.runs
else:
    NUMBER_OF_RUNS = 1

# Hidden layer
if CLUSTER:
    N_NEURONS_1 = 750 #!800 #? Should I lower it?
else:
    N_NEURONS_1 = 400
TAU_S_1 = 0.130
THRESHOLD_HAT_1 = 0.2
DELTA_THRESHOLD_1 = 1 * THRESHOLD_HAT_1
SPIKE_BUFFER_SIZE_1 = 20

# Residual layer
if CLUSTER:
    N_NEURONS_RES = 750 #!800 #? Should I lower it?
else:
    N_NEURONS_RES = 400
TAU_S_RES = 0.130
THRESHOLD_HAT_RES = 0.2
DELTA_THRESHOLD_RES = 1 * THRESHOLD_HAT_RES
SPIKE_BUFFER_SIZE_RES = 20

# Output_layer
N_OUTPUTS = 10
TAU_S_OUTPUT = 0.130
THRESHOLD_HAT_OUTPUT = 1.3
DELTA_THRESHOLD_OUTPUT = 1 * THRESHOLD_HAT_OUTPUT
SPIKE_BUFFER_SIZE_OUTPUT = 30



# Training parameters
N_TRAINING_EPOCHS = arguments.n_epochs #! used to  be 100
if CLUSTER:
    N_TRAIN_SAMPLES = 60000 #! used to be 60000
    N_TEST_SAMPLES = 10000 #! used to be 10000
    TRAIN_BATCH_SIZE = arguments.batch_size
    TEST_BATCH_SIZE = arguments.batch_size_test
else:
    N_TRAIN_SAMPLES = 6000
    N_TEST_SAMPLES = 1000
    TRAIN_BATCH_SIZE = 20
    TEST_BATCH_SIZE = 40

N_TRAIN_BATCH = int(N_TRAIN_SAMPLES / TRAIN_BATCH_SIZE)
N_TEST_BATCH = int(N_TEST_SAMPLES / TEST_BATCH_SIZE)
TRAIN_PRINT_PERIOD = 0.1
TRAIN_PRINT_PERIOD_STEP = int(N_TRAIN_SAMPLES * TRAIN_PRINT_PERIOD / TRAIN_BATCH_SIZE)
TEST_PERIOD = 1.0  # Evaluate on test batch every TEST_PERIOD epochs
TEST_PERIOD_STEP = int(N_TRAIN_SAMPLES * TEST_PERIOD / TRAIN_BATCH_SIZE)
LEARNING_RATE = arguments.learning_rate
LR_DECAY_EPOCH = int(N_TRAINING_EPOCHS/10)  # Perform decay very n epochs
LR_DECAY_FACTOR = 1.0
MIN_LEARNING_RATE = 0
TARGET_FALSE = 3
TARGET_TRUE = 30

# Plot parameters
EXPORT_METRICS = False
EXPORT_DIR = Path("./experiments/mnist/output_metrics")#+"-" + str(USE_RESIDUAL)+"-" +str(N_HIDDEN_LAYERS)+"-"+" hidden every " +str(RESIDUAL_EVERY_N) + " "+str(c) +"th Version 2")
SAVE_DIR = Path("./experiments/mnist/best_model")

#Weights and biases
# start a new wandb run to track this script



def weight_initializer(n_post: int, n_pre: int) -> cp.ndarray:
    return cp.random.uniform(-1.0, 1.0, size=(n_post, n_pre), dtype=cp.float32) # type: ignore


for run in range(NUMBER_OF_RUNS):
    print(USE_RESIDUAL, CLUSTER, FUSE_FUNCTION, N_HIDDEN_LAYERS, RESIDUAL_EVERY_N, run)
    if ALTERNATE and CLUSTER:
        USE_RESIDUAL = run%2 == 0


    
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


    
    # building the network
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
            # hidden_layer = LIFLayerResidual(previous_layer=input_layer,
            #                                 # jump_layer= hidden_layers[i-1],
            #                                 jump_layer = input_layer,
            #                                 n_neurons=N_NEURONS_RES, tau_s=TAU_S_RES,
            #                         theta=THRESHOLD_HAT_RES,
            #                         fuse_function=FUSE_FUNCTION,
            #                         delta_theta=DELTA_THRESHOLD_RES,
            #                         weight_initializer=weight_initializer,
            #                         max_n_spike=SPIKE_BUFFER_SIZE_RES,
            #                         name="Residual layer " + str(i))
            
        elif i == N_HIDDEN_LAYERS - 1 and USE_RESIDUAL and False:
            hidden_layer = LIFLayerResidual_copy(previous_layer=hidden_layers[i-1], jump_layer= hidden_layers[0], n_neurons=N_NEURONS_1, tau_s=TAU_S_RES,
                                    theta=THRESHOLD_HAT_RES,
                                    fuse_function=FUSE_FUNCTION,
                                    delta_theta=DELTA_THRESHOLD_RES,
                                    weight_initializer=weight_initializer,
                                    max_n_spike=SPIKE_BUFFER_SIZE_RES,
                                    name="Residual layer " + str(i))
        elif i % RESIDUAL_EVERY_N ==0 and USE_RESIDUAL:
            hidden_layer = LIFLayerResidual(previous_layer=hidden_layers[i-1],
                                            # jump_layer= hidden_layers[i-1],
                                            jump_layer = hidden_layers[i - RESIDUAL_JUMP_LENGTH],
                                            n_neurons=N_NEURONS_RES*2, tau_s=TAU_S_RES,
                                    theta=THRESHOLD_HAT_RES,
                                    fuse_function=FUSE_FUNCTION,
                                    delta_theta=DELTA_THRESHOLD_RES,
                                    weight_initializer=weight_initializer,
                                    max_n_spike=SPIKE_BUFFER_SIZE_RES,
                                    name="Residual layer " + str(i))
        elif i % RESIDUAL_EVERY_N ==0 and not USE_RESIDUAL:
            hidden_layer = LIFLayer(previous_layer=hidden_layers[i-1], n_neurons=int(N_NEURONS_1/2), tau_s=TAU_S_1,
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
    #End of network building

    loss_fct = SpikeCountClassLoss(target_false=TARGET_FALSE, target_true=TARGET_TRUE)
    optimizer = AdamOptimizer(learning_rate=LEARNING_RATE)
    
    for layer in network.layers:
        print(layer.name)
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
    tracker = [0.0]* len(network.layers)

    
    if USE_WANDB:
        wandb.init(
        # set the wandb project where this run will be logged
        project="Residual-SNN",
        name="Residual-SNN_"+str(N_HIDDEN_LAYERS)+"_run_"+str(run)+"verifications",
        
        # track hyperparameters and run metadata4
        config={
        "Cluster": CLUSTER,
        "FUSE_FUNCTION": FUSE_FUNCTION,
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
        "version": "2.2.0_cluster_" + str(CLUSTER),
        }
        )

    print("Training...")
    for epoch in range(N_TRAINING_EPOCHS):
        train_time_monitor.start()
        dataset.shuffle()

        # Learning rate decay
        if epoch > 0 and epoch % LR_DECAY_EPOCH == 0:
            optimizer.learning_rate = np.maximum(LR_DECAY_FACTOR * optimizer.learning_rate, MIN_LEARNING_RATE) # type: ignore

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
            #! the second set of gradients form the residual layer has nans
            gradient = network.backward(errors)
            # avg_gradient = [None if g is None else cp.mean(g, axis=0) for g, layer in zip(gradient, network.layers)]
            
            avg_gradient = []

            for g, layer in zip(gradient, network.layers):
                if g is None:
                    avg_gradient.append(None)
                elif isinstance(layer, LIFLayerResidual) and FUSE_FUNCTION == "Append":
                    grad_entry = []
                    for i in range(len(g)):
                        try:
                            averaged_values = cp.mean(g[i], axis=0)
                        except:
                            averaged_values = cp.mean(g[0], axis=0)
                        grad_entry.append(averaged_values)
                    avg_gradient.append(grad_entry)
                else:
                    averaged_values = cp.mean(g, axis=0)
                    avg_gradient.append(averaged_values)

            training_steps += 1
            
            #gradient 
            del gradient
            # keep track of the gradient
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
            #! problem is here, in has no nans but deltas has nans
            deltas = optimizer.step(avg_gradient)
            del avg_gradient
            #! deltas have nans
            network.apply_deltas(deltas)
            del deltas

            epoch_metrics = training_steps * TRAIN_BATCH_SIZE / N_TRAIN_SAMPLES

            # Training metrics
            if training_steps % TRAIN_PRINT_PERIOD_STEP == 0:
                # Compute metrics

                train_monitors_manager.record(epoch_metrics)
                train_monitors_manager.print(epoch_metrics, use_wandb=USE_WANDB)
                train_monitors_manager.export()
                out_spikes, n_out_spikes = network.output_spike_trains
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

                for l, mon in test_norm_monitors.items():
                    #TODO what does this do?
                    # moni
                    if type(l.weights) is tuple:
                        mon.add(l.weights[0])
                        mon.add(l.weights[1])
                    else:
                        mon.add(l.weights) # type: ignore

                test_learning_rate_monitor.add(optimizer.learning_rate) # type: ignore

                records = test_monitors_manager.record(epoch_metrics)
                test_monitors_manager.print(epoch_metrics, use_wandb=USE_WANDB)
                test_monitors_manager.export()

                acc = records[test_accuracy_monitor]
                loss_to_save = records[test_loss_monitor]
                

                mean_res = cp.mean(cp.array(mean_spikes_for_times))
                mean_first = cp.mean(cp.array(first_spike_for_times))
                if not CLUSTER:
                    print(f'Output layer mean times: {mean_res}')
                    print(f'Output layer first spike: {mean_first}')
                if USE_WANDB:
                    wandb.log({"Test_mean_spikes_for_times": float(mean_res), "Test_first_spike_for_times": float(mean_first)})

                if acc > best_acc:
                    best_acc = acc
                    # network.store(SAVE_DIR)
                    print(f"Best accuracy: {np.around(best_acc, 2)}%, Networks save NOT to: {SAVE_DIR}")
    # best_acc_array.append(best_acc)    
        
    # with open('times.txt', 'a') as f:
    #     string =f'End of run: {c}'+ "\n"
    #     f.write(string)
    if USE_WANDB:
        wandb.finish()
    print("Done!: ", run)


# wandb.finish()

# Write average accuracy to file
# avg_acc = np.mean(best_acc_array)
# print("Average accuracy: ", avg_acc)
# with open('avg_acc.txt', 'a') as f:
#     string = "With # of hidden layers: "+str(N_HIDDEN_LAYERS)+"\n"+ "Residual: "+str(USE_RESIDUAL)+"\n"+ "Residual every: "+str(RESIDUAL_EVERY_N)+"\n"+"Average accuracy: " +  str(avg_acc) + "\n" + "Accuracies:" + str(best_acc_array)+"\n" +"-------------------------------------"+"\n"
#     f.write(string)
print("Done!")
