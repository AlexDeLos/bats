from re import L
import re
from typing import Optional, List
import cupy as cp
import numpy as np

from ..AbstractOptimizer import AbstractOptimizer

update_m_kernel = cp.ElementwiseKernel("float32 m, float32 beta_1, float32 one_minus_beta_1, float32 grad",
                                       "float32 new_m",
                                       "new_m = beta_1 * m + one_minus_beta_1 * grad",
                                       "update_m_kernel")
update_v_kernel = cp.ElementwiseKernel("float32 v, float32 beta_2, float32 one_minus_beta_2, float32 grad",
                                       "float32 new_v",
                                       "new_v = beta_2 * v + one_minus_beta_2 * grad * grad",
                                       "update_v_kernel")
compute_deltas_kernel = cp.ElementwiseKernel("float32 grad, float32 m_hat, float32 v_hat, float32 learning_rate,"
                                             "float32 epsilon",
                                             "float32 delta",
                                             "delta = -(learning_rate * m_hat / (sqrtf(v_hat) + epsilon))",
                                             "compute_deltas_kernel")


class AdamOptimizer(AbstractOptimizer):
    def __init__(self, beta_1=0.9, beta_2=0.999, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.__beta_1: cp.float32 = cp.float32(beta_1)
        self.__one_minus_beta_1: cp.float32 = cp.float32(1.0 - beta_1)
        self.__beta_2: cp.float32 = cp.float32(beta_2)
        self.__one_minus_beta_2: cp.float32 = cp.float32(1.0 - beta_2)
        self.__epsilon: cp.float32 = cp.float32(epsilon)

        self.__m: Optional[List[List[cp.array]]] = None # type: ignore
        self.__v: Optional[List[List[cp.array]]] = None # type: ignore
        self.__t: cp.int32 = cp.int32(0)

    def step(self, gradient: List[cp.ndarray]) -> List[cp.ndarray]:
        self.__t += 1 # type: ignore

        # Set m and v to 0 at first iteration
        if self.__m is None:
            #! Error found here: 2 (found here when no break points are set)
            self.__m = []
            self.__v = []
            for grad in gradient:
                if grad is None:
                    self.__m.append(None)
                    self.__v.append(None)
                elif type(grad) == list:
                    #! this needs to be changed, we are only using the first element of the list, not both as we should?
                    touple = [cp.zeros(grad[0].shape, dtype=cp.float32), cp.zeros(grad[1].shape, dtype=cp.float32)]
                    self.__m.append(touple)
                    self.__v.append(touple)
                else:
                    self.__m.append(cp.zeros(grad.shape, dtype=cp.float32))
                    self.__v.append(cp.zeros(grad.shape, dtype=cp.float32))
            # self.__m = [None if grad is None else cp.zeros(grad.shape, dtype=cp.float32) for grad in gradient] # type: ignore
            # self.__v = [None if grad is None else cp.zeros(grad.shape, dtype=cp.float32) for grad in gradient] # type: ignore
        # Update m and v
        # m = self.__m
        # for pre_m, grad in zip(self.__m, gradient):
        #     if pre_m is None:
        #         continue
        #     if type(grad) == list:
        #         pre_m0 = pre_m[0]
        #         grad0 = grad[0]
        #         sol1 = update_m_kernel(pre_m0, self.__beta_1, self.__one_minus_beta_1, grad0)
        #         pre_m[0] = sol1
        #         pre_m1 = pre_m[1]
        #         grad1 = grad[1]
        #         sol2 = update_m_kernel(pre_m1, self.__beta_1, self.__one_minus_beta_1, grad1)
        #         pre_m[1] = sol2
        #     else:
        #         pre_m = update_m_kernel(pre_m, self.__beta_1, self.__one_minus_beta_1, grad)
        self.__m = [None if grad is None else # type: ignore
                    #! care that this is hard coded for jut two elements
                    (update_m_kernel(pre_m[0], self.__beta_1, self.__one_minus_beta_1, grad[0]),
                     update_m_kernel(pre_m[1], self.__beta_1, self.__one_minus_beta_1, grad[1])) if type(grad) == list
                    else
                    update_m_kernel(pre_m, self.__beta_1, self.__one_minus_beta_1, grad)
                    for pre_m, grad in zip(self.__m, gradient)]
        test1 = self.__m
        self.__v = [None if grad is None else
                    #! care that this is hard coded for jut two elements
                    (update_v_kernel(pre_v[0], self.__beta_2, self.__one_minus_beta_2, grad[0]), update_v_kernel(pre_v[1], self.__beta_2, self.__one_minus_beta_2, grad[1])) if type(grad) == list # type: ignore
                    else
                    update_v_kernel(pre_v, self.__beta_2, self.__one_minus_beta_2, grad)
                    for pre_v, grad in zip(self.__v, gradient)] # type: ignore
        test2 = self.__v

        # Compute m_hat and v_hat
        one_minus_beta_1_power_t = 1 - self.__beta_1 ** self.__t
        one_minus_beta_2_power_t = 1 - self.__beta_2 ** self.__t
        # TODO: Implicit conversion to a NumPy array is not allowed. Please use `.get()` to construct a NumPy array explicitly.
        m_hat = [None if m is None else 
                 (m[0] / one_minus_beta_1_power_t, m[1] / one_minus_beta_1_power_t)  if type(m) == tuple 
                else
                 m / one_minus_beta_1_power_t 
                 for m in self.__m]
        v_hat = [None if v is None else
                 (v[0] / one_minus_beta_2_power_t,v[1] / one_minus_beta_2_power_t) if type(v) == tuple
                 else
                 v / one_minus_beta_2_power_t for v in self.__v]

        to_ret = []
        for g,m,v in zip(gradient, m_hat, v_hat):
            if g is None:
                to_ret.append(None)
            elif type(g) == list:
                touple1 = compute_deltas_kernel(g[0], m[0], v[0], self._learning_rate, self.__epsilon)
                touple2 = compute_deltas_kernel(g[1], m[1], v[1], self._learning_rate, self.__epsilon)
                to_ret.append((touple1, touple2))
            else:
                to_ret.append(compute_deltas_kernel(g, m, v, self._learning_rate, self.__epsilon))
        # to_ret2 =  [None 
        #         if g is None
        #         else
        #         (compute_deltas_kernel(g[0], m[0], v[0], self._learning_rate, self.__epsilon), compute_deltas_kernel(g[1], m[1], v[1], self._learning_rate, self.__epsilon))
        #         if type(g) == list
        #         else
        #         compute_deltas_kernel(g, m, v, self._learning_rate, self.__epsilon)
        #         for g, m, v in zip(gradient, m_hat, v_hat)]
        return to_ret
