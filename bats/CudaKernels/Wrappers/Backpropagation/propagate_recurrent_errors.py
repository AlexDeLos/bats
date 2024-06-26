from bats.CudaKernels.load_kernel import load_kernel
import cupy as cp

KERNEL_FILE = "Backpropagation/propagate_recurrent_errors.cu"
KERNEL_NAME = "propagate_recurrent_errors_kernel"

__propagate_recurrent_errors_kernel = load_kernel(KERNEL_FILE, KERNEL_NAME)

def propagate_recurrent_errors(x: cp.ndarray, exp_tau: cp.ndarray, errors: cp.ndarray,
                               delta_theta_tau: cp.float32, residual: bool = False) -> None:

    batch_size, n_neurons, max_n_spike = errors.shape
    batch_size, n_neurons2, max_n_spike = x.shape

    block_dim = (batch_size, 1, 1)
    grid_dim = (n_neurons, 1, 1)
    __propagate_recurrent_errors_kernel(grid_dim, block_dim, (x, exp_tau, errors, delta_theta_tau,
                                                              cp.int32(max_n_spike)))
    # if cp.any(cp.isnan(errors)):
    #     arr_clean = errors[~cp.isnan(errors)]
    #     arr_clean= cp.reshape(arr_clean, (batch_size,n_neurons2,max_n_spike))
    #     raise ValueError("NaNs in errors")
    
    