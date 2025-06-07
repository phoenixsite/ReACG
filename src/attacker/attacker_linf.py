import torch

from attacker.auto_conjugate_attack import ACG
from attacker.auto_projected_gradient_attack import APGD
from attacker.rescaling_auto_conjugate_gradient_attack import ReACG
from core.projection import ProjectionLinf
from utils import setup_logger

logger = setup_logger(__name__)

class ACGLinf(ACG):
    def __init__(self, *args, **kwargs):
        super(ACGLinf, self).__init__(*args, **kwargs)
        self.lower: torch.Tensor = None
        self.upper: torch.Tensor = None

    @torch.no_grad()
    def set_bounds(self, x_nat):
        self.upper = (x_nat + self.epsilon).clamp(0, 1).clone().to(self.device)
        self.lower = (x_nat - self.epsilon).clamp(0, 1).clone().to(self.device)
        assert isinstance(self.lower, torch.Tensor)

    @torch.no_grad()
    def set_projection(self, x_nat: torch.Tensor):
        self.set_bounds(x_nat)
        self.projection = ProjectionLinf(lower=self.lower, upper=self.upper)

    def check_feasibility(self, x: torch.Tensor):
        bad_values = [value for value in x if value < self.lower.cpu()]

        if bad_values.any():
            raise ValueError(f"There is some value in x lower than the lower bound: lower bound={self.lower}, bad_values={bad_values}")
    
        bad_values = [value for value in x if value > self.upper.cpu()]
        if bad_values.any():
            raise ValueError(f"There is some value in x greater than the upper bound: upper bound={self.upper}, bad_values={bad_values}")


class ReACGLinf(ReACG):
    def __init__(self, *args, **kwargs):
        super(ReACGLinf, self).__init__(*args, **kwargs)
        self.lower: torch.Tensor = None
        self.upper: torch.Tensor = None

    @torch.no_grad()
    def set_bounds(self, x_nat):
        self.upper = (x_nat + self.epsilon).clamp(0, 1).clone().to(self.device)
        self.lower = (x_nat - self.epsilon).clamp(0, 1).clone().to(self.device)
        assert isinstance(self.lower, torch.Tensor)

    @torch.no_grad()
    def set_projection(self, x_nat: torch.Tensor):
        self.set_bounds(x_nat)
        self.projection = ProjectionLinf(lower=self.lower, upper=self.upper)

    def check_feasibility(self, x: torch.Tensor):
        bad_values = (x < self.lower.cpu())

        if bad_values.any():
            raise ValueError(f"There is some value in x lower than the lower bound: lower bound={self.lower}, bad_values={bad_values}")
    
        bad_values = (x > self.upper.cpu())
        if bad_values.any():
            raise ValueError(f"There is some value in x greater than the upper bound: upper bound={self.upper}, bad_values={bad_values}")


class APGDLinf(APGD):
    def __init__(self, *args, **kwargs):
        super(APGDLinf, self).__init__(*args, **kwargs)
        self.lower: torch.Tensor = None
        self.upper: torch.Tensor = None

    @torch.no_grad()
    def set_bounds(self, x_nat):
        self.upper = (x_nat + self.epsilon).clamp(0, 1).clone().to(self.device)
        self.lower = (x_nat - self.epsilon).clamp(0, 1).clone().to(self.device)
        assert isinstance(self.lower, torch.Tensor)

    @torch.no_grad()
    def set_projection(self, x_nat: torch.Tensor):
        self.set_bounds(x_nat)
        self.projection = ProjectionLinf(lower=self.lower, upper=self.upper)

    def check_feasibility(self, x: torch.Tensor):

        bad_values = (x < self.lower.cpu())

        if bad_values.any():
            raise ValueError(f"There is some value in x lower than the lower bound: lower bound={self.lower}, bad_values={bad_values}")
    
        bad_values = (x > self.upper.cpu())
        if bad_values.any():
            raise ValueError(f"There is some value in x greater than the upper bound: upper bound={self.upper}, bad_values={bad_values}")
