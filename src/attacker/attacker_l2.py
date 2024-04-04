import torch

from attacker.auto_conjugate_attack import ACG
from attacker.auto_projected_gradient_attack import APGD
from attacker.rescaling_auto_conjugate_gradient_attack import ReACG
from core.projection import ProjectionL2
from utils import setup_logger

logger = setup_logger(__name__)


class ACGL2(ACG):
    def __init__(self, *args, **kwargs):
        super(ACGL2, self).__init__(*args, **kwargs)

    def set_projection(self, x_nat: torch.Tensor):
        self.x_nat = x_nat.clone()
        self.projection = ProjectionL2(
            epsilon=self.epsilon, x_nat=x_nat, _min=0.0, _max=1.0
        )

    def check_feasibility(self, x: torch.Tensor):
        assert (
            (x - self.x_nat.cpu()).norm(p=2, dim=(1, 2, 3)) <= self.epsilon + 1e-5
        ).all()
        assert (x >= 0.0).all()
        assert (x <= 1.0).all()


class ReACGL2(ReACG):
    def __init__(self, *args, **kwargs):
        super(ReACGL2, self).__init__(*args, **kwargs)

    def set_projection(self, x_nat: torch.Tensor):
        self.x_nat = x_nat.clone()
        self.projection = ProjectionL2(
            epsilon=self.epsilon, x_nat=x_nat, _min=0.0, _max=1.0
        )

    def check_feasibility(self, x: torch.Tensor):
        assert (
            (x - self.x_nat.cpu()).norm(p=2, dim=(1, 2, 3)) <= self.epsilon + 1e-5
        ).all()
        assert (x >= 0.0).all()
        assert (x <= 1.0).all()


class APGDL2(APGD):
    def __init__(self, *args, **kwargs):
        super(APGDL2, self).__init__(*args, **kwargs)

    def set_projection(self, x_nat: torch.Tensor):
        self.x_nat = x_nat.clone()
        self.projection = ProjectionL2(
            epsilon=self.epsilon, x_nat=x_nat, _min=0.0, _max=1.0
        )

    def check_feasibility(self, x: torch.Tensor):
        assert (
            (x - self.x_nat.cpu()).norm(p=2, dim=(1, 2, 3)) <= self.epsilon + 1e-5
        ).all()
        assert (x >= 0.0).all()
        assert (x <= 1.0).all()
