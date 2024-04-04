from abc import ABCMeta, abstractmethod
from typing import Callable, Dict, Union

import torch

from core.criterion import Criterion, CriterionManager
from core.projection import Projection
from utils import setup_logger

logger = setup_logger(__name__)


class BaseAttacker(metaclass=ABCMeta):
    """White-box Attack"""

    def __init__(self, epsilon: float, device: Union[str, torch.device]) -> None:
        self.epsilon: float = epsilon
        self.device: Union[str, torch.device] = device
        self.projection: Projection = None
        self.nam: str = None

    @abstractmethod
    def attack(
        self,
        x_nat: torch.Tensor,
        y_true: torch.Tensor,
        parameters: Dict,
        criterion: Union[Criterion, CriterionManager],
        get_initialpoint: Callable,
        n_forward: int,
        n_backward: int,
        *args,
        **kwargs
    ):
        pass
    
    @abstractmethod
    def set_name(self, parameters: Dict):
        pass

    @torch.no_grad()
    def set_projection(self, x: torch.Tensor) -> None:
        self.projection = lambda x: x.clamp(min=0.0, max=1.0)

    @torch.no_grad()
    def check_feasibility(self, x: torch.Tensor) -> None:
        assert (x >= 0.0).all()
        assert (x <= 1.0).all()
