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
        assert (x >= self.lower.cpu()).all()
        assert (x <= self.upper.cpu()).all()


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
        assert (x >= self.lower.cpu()).all()
        assert (x <= self.upper.cpu()).all()


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
        assert (x >= self.lower.cpu()).all()
        assert (x <= self.upper.cpu()).all()
