from ast import Tuple
from typing import List, Dict
import numpy as np
import cupy as cp
import wandb

from bats.AbstractMonitor import AbstractMonitor
from bats.Monitors.SilentNeuronsMonitor import SilentNeuronsMonitor


class MonitorsManager:
    def __init__(self, monitors: List[AbstractMonitor] = [], print_prefix: str = ""):
        self._monitors: List[AbstractMonitor] = monitors
        self._print_prefix = print_prefix

    def add(self, monitor: AbstractMonitor) -> None:
        self._monitors.append(monitor)

    def print(self, epoch: float, decimal: int = 1, use_wandb = False, w_b = None) -> None:
        returns = []
        to_print = self._print_prefix + f"Epoch {np.around(epoch, decimal)}"
        silent_avg = 0
        silent_count = 0
        for i, monitor in enumerate(self._monitors):
            if isinstance(monitor, SilentNeuronsMonitor):
                silent_avg = monitor._values[-1]
                silent_count = silent_count + 1
                continue
            to_print += (" | " if i == 0 else ", ") + str(monitor)
            if use_wandb:
                value = monitor._values[-1]
                if type(value) == cp.ndarray:
                    value = float(value)
                w_b.save({self._print_prefix + monitor._name: value})
            returns.append((monitor._name, monitor._values[-1]))
        if silent_count > 0:
            to_print += f" | Silent labels: {silent_avg:.2f}%"
            if use_wandb:
                w_b.save({self._print_prefix + "Silent neurons (%)": silent_avg})
        print(to_print)

    def record(self, epoch: float) -> Dict:
        return {monitor: monitor.record(epoch) for monitor in self._monitors}

    def export(self) -> None:
        for monitor in self._monitors:
            monitor.export()