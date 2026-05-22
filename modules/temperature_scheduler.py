import torch
import math
from typing import Optional, Union
from abc import ABC, abstractmethod


class TemperatureScheduler(ABC):
    def __init__(
        self,
        initial_temperature: float = 2.0,
        min_temperature: float = 0.1,
        max_temperature: Optional[float] = None
    ):
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature or initial_temperature
        self.current_temperature = initial_temperature
        self.step_count = 0
    
    @abstractmethod
    def step(self) -> float:
        pass

    def reset(self) -> None:
        self.current_temperature = self.initial_temperature
        self.step_count = 0

    def get_temperature(self) -> float:
        return self.current_temperature


class ExponentialScheduler(TemperatureScheduler):
    """Exponential decay: \tau(t) = max(\tau_min, \tau_0 * decay^t)"""

    def __init__(
        self,
        initial_temperature: float = 2.0,
        min_temperature: float = 0.1,
        decay_rate: float = 0.999,
        max_temperature: Optional[float] = None,
        **kwargs
    ):
        super().__init__(initial_temperature, min_temperature, max_temperature)
        self.decay_rate = decay_rate
    
    def step(self) -> float:
        self.step_count += 1
        self.current_temperature = max(
            self.min_temperature,
            self.current_temperature * self.decay_rate
        )
        return self.current_temperature


class CosineScheduler(TemperatureScheduler):
    """Cosine annealing: smooth transition from max to min temperature."""

    def __init__(
        self,
        initial_temperature: float = 2.0,
        min_temperature: float = 0.1,
        total_steps: int = 10000,
        max_temperature: Optional[float] = None,
        **kwargs
    ):
        super().__init__(initial_temperature, min_temperature, max_temperature)
        self.total_steps = total_steps
    
    def step(self) -> float:
        self.step_count += 1
        progress = min(self.step_count / self.total_steps, 1.0)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        self.current_temperature = (
            self.min_temperature + 
            (self.max_temperature - self.min_temperature) * cosine_factor
        )
        return self.current_temperature
    
    def set_total_steps(self, total_steps: int) -> None:
        self.total_steps = total_steps


class InverseLogScheduler(TemperatureScheduler):
    """Inverse logarithmic: \tau(t) = \tau_min + (\tau_max - \tau_min) / (1 + \alpha * log(1 + t))"""

    def __init__(
        self,
        initial_temperature: float = 2.0,
        min_temperature: float = 0.1,
        log_rate: float = 0.1,
        max_temperature: Optional[float] = None,
        **kwargs
    ):
        super().__init__(initial_temperature, min_temperature, max_temperature)
        self.log_rate = log_rate
    
    def step(self) -> float:
        self.step_count += 1
        log_factor = 1 + self.log_rate * math.log(1 + self.step_count)
        self.current_temperature = (
            self.min_temperature + 
            (self.max_temperature - self.min_temperature) / log_factor
        )
        return self.current_temperature


class PowerLawScheduler(TemperatureScheduler):
    """Power law decay: \tau(t) = max(\tau_min, \tau_max * (t + 1)^(-β))"""

    def __init__(
        self,
        initial_temperature: float = 2.0,
        min_temperature: float = 0.1,
        beta: float = 0.5,
        max_temperature: Optional[float] = None,
        **kwargs
    ):
        super().__init__(initial_temperature, min_temperature, max_temperature)
        self.beta = beta
    
    def step(self) -> float:
        self.step_count += 1
        power_factor = (self.step_count + 1) ** (-self.beta)
        self.current_temperature = max(
            self.min_temperature,
            self.max_temperature * power_factor
        )
        return self.current_temperature


class ConstantScheduler(TemperatureScheduler):
    """Constant temperature (no annealing)."""
    
    def step(self) -> float:
        return self.current_temperature


def create_temperature_scheduler(
    schedule_type: str,
    initial_temperature: float = 2.0,
    min_temperature: float = 0.1,
    **kwargs
) -> TemperatureScheduler:
    schedulers = {
        "exponential": ExponentialScheduler,
        "cosine": CosineScheduler,
        "inverse_log": InverseLogScheduler,
        "power_law": PowerLawScheduler,
        "constant": ConstantScheduler,
    }
    
    # remove decay rate if it is constant scheduler
    if schedule_type == "constant":
        kwargs.pop("decay_rate", None)
        kwargs.pop("total_steps", None)
        kwargs.pop("log_rate", None)
        kwargs.pop("beta", None)
    
    if schedule_type not in schedulers:
        raise ValueError(f"Unknown schedule type: {schedule_type}. Available: {list(schedulers.keys())}")
    
    return schedulers[schedule_type](
        initial_temperature=initial_temperature,
        min_temperature=min_temperature,
        **kwargs
    )
