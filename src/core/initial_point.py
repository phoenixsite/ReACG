from typing import Dict, Union

import torch

from core.criterion import Criterion, CriterionManager
from core.projection import Projection
from utils import setup_logger
from utils.helper_functions import normalize

logger = setup_logger(__name__)


def get_naive_initialpoint(
    x_nat: torch.Tensor,
    epsilon: float,
    projection: Projection,
    parameters: Dict,
    n_forward: int,
    n_backward: int,
    *args,
    **kwargs,
):
    initialpoint_method = parameters["initial_point"]
    begin = 0
    if initialpoint_method == "input":
        xk = x_nat.clone()
    elif initialpoint_method == "random":
        xk = projection(x_nat + torch.empty_like(x_nat).uniform_(-epsilon, epsilon))
    elif initialpoint_method == "bernoulli":
        noise = torch.empty_like(x_nat).uniform_(-1.0, 1.0)
        noise[noise > 0] = 1.0
        noise[noise < 0] = -1.0
        xk = projection(x_nat + epsilon * noise)
    elif initialpoint_method == "center":
        xk = (projection.lower + projection.upper) / 2
    return xk, begin, n_forward, n_backward


class OutputDiversifiedSampling:
    def __init__(self, output_dim: int):
        self.output_dim = output_dim

    def __call__(
        self,
        x_nat: torch.Tensor,
        y_true: torch.Tensor,
        projection: Projection,
        criterion: Union[Criterion, CriterionManager],
        parameters: Dict,
        n_forward: int,
        n_backward: int,
        *args,
        **kwargs,
    ):
        bs = x_nat.shape[0]
        odi_iter = parameters["odi_iter"]
        odi_step = parameters["odi_step"]

        # FIXME: fix seed
        w = torch.empty((bs, self.output_dim), device=x_nat.device).uniform_(-1, 1)

        x = x_nat.clone()
        for i in range(odi_iter):
            criterion_outs = criterion(
                x, y_true, w=w, criterion_name="ods", enable_grad=True
            )
            n_forward += 1
            n_backward += 1
            vods = normalize(criterion_outs.grad, "vods")
            x = projection(x + odi_step * torch.sign(vods))
            logger.debug(
                f"[ ODS iteration {i} ]: cw loss = {criterion_outs.cw_loss.mean().item():.4f}"
            )
        return x.detach().clone(), odi_iter, n_forward, n_backward
