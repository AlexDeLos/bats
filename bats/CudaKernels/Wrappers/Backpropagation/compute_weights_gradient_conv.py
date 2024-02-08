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

    errors = cp.zeros(errors.shape, dtype=cp.float32)
    f1 = cp.zeros(f1.shape, dtype=cp.float32)
    f2 = cp.zeros(f2.shape, dtype=cp.float32)
    pre_times = cp.zeros(pre_times.shape, dtype=cp.float32)
    post_times = cp.zeros(post_times.shape, dtype=cp.float32)
    pre_exp_tau_s = cp.zeros(pre_exp_tau_s.shape, dtype=cp.float32)
    pre_exp_tau = cp.zeros(pre_exp_tau.shape, dtype=cp.float32)


    filter_c, filter_x, filter_y, filter_z = filter_shape.get()
    gradient = cp.zeros((batch_size, filter_c, filter_x, filter_y, filter_z), dtype=cp.float32)

    block_dim = (batch_size, 1, 1)
    grid_dim = (filter_x, filter_y, filter_z)
    __compute_weights_gradient_conv_kernel(grid_dim, block_dim, (f1, f2, post_times, pre_times,
                                                                 pre_exp_tau_s, pre_exp_tau, errors,
                                                                 pre_shape, post_shape, filter_shape,
                                                                 gradient,
                                                                 n_post_neurons, n_pre_neurons,
                                                                 cp.int32(max_n_post_spike),
                                                                 cp.int32(max_n_pre_spike)))
    if cp.any(cp.isnan(gradient)):
        #! maybe something with the max spikes?
        #* errors are not the problem
        num_nans_grad = cp.sum(cp.isnan(gradient))
        ups = True
    return gradient
