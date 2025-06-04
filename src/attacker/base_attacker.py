from abc import ABCMeta, abstractmethod
from typing import Callable, Dict, Union

import torch

from core.criterion import Criterion, CriterionManager
from core.projection import Projection
from utils import setup_logger

logger = setup_logger(__name__)


class BaseAttacker(metaclass=ABCMeta):
    """
    White-box Attack
    
    :param float epsilon: Supreme distance between the adversarial example
        and the original image.
    :param Union[str, torch.device] device: GPU or GPU to make the operations.
    :param Projection projection: Function used to project the modifications
        of an image to the feasible region.
    :param str name: Name of the attack. Depends on the parameters.
    """

    def __init__(self, params: Dict, device: Union[str, torch.device]) -> None:
        """
        Set the epsilon, the device where to run the operations and 
        the name of the attacker.
        """

        self.epsilon: float = params.epsilon
        self.device: Union[str, torch.device] = device
        self.projection: Projection = None
        self.name: str = self._set_name(params)

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

    @torch.no_grad()
    def set_projection(self) -> None:
        self.projection = lambda x: x.clamp(min=0.0, max=1.0)

    @torch.no_grad()
    def check_feasibility(self, x: torch.Tensor) -> None:
        assert (x >= 0.0).all()
        assert (x <= 1.0).all()
