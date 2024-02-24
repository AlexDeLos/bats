from calendar import c
import cupy as cp

from bats.CudaKernels.load_kernel import load_kernel

KERNEL_FILE = "Backpropagation/compute_weights_gradient_conv.cu"
KERNEL_NAME = "compute_weights_gradient_conv_kernel"

__compute_weights_gradient_conv_kernel = load_kernel(KERNEL_FILE, KERNEL_NAME)


def compute_weights_gradient_conv(f1: cp.ndarray, f2: cp.ndarray,
                                  post_times: cp.ndarray, pre_times: cp.array,
                                  pre_exp_tau_s: cp.ndarray, pre_exp_tau: cp.ndarray,
                                  pre_shape: cp.ndarray, post_shape: cp.ndarray,
                                  filter_shape: cp.ndarray,
                                  errors: cp.ndarray) -> cp.ndarray:
    batch_size, n_post_neurons, max_n_post_spike = f1.shape
    _, n_pre_neurons, max_n_pre_spike = pre_exp_tau_s.shape

    # errors = cp.zeros(errors.shape, dtype=cp.float32)
    # f1 = cp.zeros(f1.shape, dtype=cp.float32)
    # f2 = cp.zeros(f2.shape, dtype=cp.float32)
    # pre_times = cp.zeros(pre_times.shape, dtype=cp.float32)
    # post_times = cp.zeros(post_times.shape, dtype=cp.float32)
    # pre_exp_tau_s = cp.zeros(pre_exp_tau_s.shape, dtype=cp.float32)
    # pre_exp_tau = cp.zeros(pre_exp_tau.shape, dtype=cp.float32)


    filter_c, filter_x, filter_y, filter_z = filter_shape.get()
    gradient = cp.zeros((batch_size, filter_c, filter_x, filter_y, filter_z), dtype=cp.float32)

    block_dim = (batch_size, 1, 1)
    grid_dim = (filter_x, filter_y, filter_z)
    # Variables for debugging

    # test_f1 = f1.copy()
    # test_f2 = f2.copy()
    # test_post_times = post_times.copy()
    # test_pre_times = pre_times.copy()
    # test_pre_exp_tau_s = pre_exp_tau_s.copy()
    # test_pre_exp_tau = pre_exp_tau.copy()
    # test_errors = errors.copy()
    # test_pre_shape = pre_shape.copy()
    # test_post_shape = post_shape.copy()
    # test_filter_shape = filter_shape.copy()
    og_grad = gradient.copy()
    # test_n_post_neurons = n_post_neurons
    # test_n_pre_neurons = n_pre_neurons
    # test_max_n_post_spike = max_n_post_spike
    # test_max_n_pre_spike = max_n_pre_spike
    any_nans = cp.any(cp.isnan(f1)) or cp.any(cp.isnan(f2)) or cp.any(cp.isnan(post_times)) or cp.any(cp.isnan(pre_times)) or cp.any(cp.isnan(pre_exp_tau_s)) or cp.any(cp.isnan(pre_exp_tau))# or cp.any(cp.isnan(errors))
    __compute_weights_gradient_conv_kernel(grid_dim, block_dim, (f1, f2, post_times, pre_times,
                                                                 pre_exp_tau_s, pre_exp_tau, errors,
                                                                 pre_shape, post_shape, filter_shape,
                                                                 gradient,
                                                                 n_post_neurons, n_pre_neurons,
                                                                 cp.int32(max_n_post_spike),
                                                                 cp.int32(max_n_pre_spike)))
    

    #* propably a memory error, check for buffers
    #! OK, how do I now make sire the new paddings don't mess up the gradient?
    if cp.any(cp.isnan(gradient)):
    #     #! maybe something with the max spikes?
    #     #* errors are not the problem
        print(f"Found nans in gradient", cp.where(cp.isnan(gradient)))
    #     num_nans_grad = cp.sum(cp.isnan(gradient))
    #     # with all of shape [3,3,1] I get 3 nan values
        out_grad = gradient.copy()
        count = 0
        while cp.any(cp.isnan(out_grad)):
            out_grad = og_grad.copy()
            count += 1
            __compute_weights_gradient_conv_kernel(grid_dim, block_dim, (f1, f2, post_times, pre_times,
                                                                        pre_exp_tau_s, pre_exp_tau, errors,
                                                                        pre_shape, post_shape, filter_shape,
                                                                        out_grad,
                                                                        n_post_neurons, n_pre_neurons,
                                                                        cp.int32(max_n_post_spike),
                                                                        cp.int32(max_n_pre_spike)))
            print(f"Found nans in gradient,trying again: ", count, cp.where(cp.isnan(out_grad)))
        #! problem could be with the channels
    
    return gradient
