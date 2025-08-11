"""
Temperature scheduling for Gumbel Softmax quantization.

This module provides various temperature annealing strategies that can be used
during training to gradually transition from soft to hard assignments.
"""

import torch
import math
from typing import Optional, Union
from abc import ABC, abstractmethod


class TemperatureScheduler(ABC):
    """Base class for temperature scheduling strategies."""
    
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
        """Update temperature and return new value."""
        pass
    
    def reset(self) -> None:
        """Reset scheduler to initial state."""
        self.current_temperature = self.initial_temperature
        self.step_count = 0
    
    def get_temperature(self) -> float:
        """Get current temperature value."""
        return self.current_temperature


class ExponentialScheduler(TemperatureScheduler):
    """Exponential decay: τ(t) = max(τ_min, τ_0 * decay^t)"""

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
        """Update total steps for the schedule."""
        self.total_steps = total_steps


class InverseLogScheduler(TemperatureScheduler):
    """Inverse logarithmic: τ(t) = τ_min + (τ_max - τ_min) / (1 + α * log(1 + t))"""

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
    """Power law decay: τ(t) = max(τ_min, τ_max * (t + 1)^(-β))"""

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
    """Factory function to create temperature schedulers."""
    
    schedulers = {
        "exponential": ExponentialScheduler,
        "cosine": CosineScheduler,
        "inverse_log": InverseLogScheduler,
        "power_law": PowerLawScheduler,
        "constant": ConstantScheduler,
    }
    
    if schedule_type not in schedulers:
        raise ValueError(f"Unknown schedule type: {schedule_type}. Available: {list(schedulers.keys())}")
    
    return schedulers[schedule_type](
        initial_temperature=initial_temperature,
        min_temperature=min_temperature,
        **kwargs
    )
