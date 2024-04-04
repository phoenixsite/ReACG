from typing import Dict

import torch

# from utils.logging import setup_logger

# logger = setup_logger(__name__)


class BaseDict(Dict):
    """simple implementation of attribute dict"""

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.__dict__ = self

    def copy(self, *args, **kwargs):
        _obj = super(BaseDict, self).copy()
        new_obj = BaseDict()
        new_obj.update(_obj)
        return new_obj


class CriterionOuts(BaseDict):
    """Stores the output of the criterion.

    Attributes
    ----------
    loss : torch.Tensor
        Objective values.
    cw_loss : torch.Tensor
        CW loss values.
    softmax_cw_loss : torch.Tensor
        Loss values of the CW loss scaled by softmax function.
    grad : torch.Tensor
        Gradient of the objective function.
    target_class : torch.Tensor
        Class label with the 2nd highest classification probability.
    logit : torch.Tensor
        Logit (row output of the classification model)
    """

    def __init__(
        self,
        loss=None,
        cw_loss=None,
        softmax_cw_loss=None,
        grad=None,
        target_class=None,
        logit=None,
        *args,
        **kwargs,
    ) -> None:
        super(CriterionOuts, self).__init__(
            loss=loss,
            cw_loss=cw_loss,
            softmax_cw_loss=softmax_cw_loss,
            grad=grad,
            target_class=target_class,
            logit=logit,
            *args,
            **kwargs,
        )


@torch.no_grad()
def to_cpu(criterion_outs):
    """Create new instance on cpu

    Parameters
    ----------
    criterion_outs : CriterionOuts
        CriterionOuts instance to be send to cpu.

    Returns
    -------
    CriterionOuts
        New instance on cpu.
    """
    _grad = (
        criterion_outs.grad.detach().clone().cpu()
        if isinstance(criterion_outs.grad, torch.Tensor)
        else criterion_outs.grad
    )
    criterion_outs_cpu = CriterionOuts(
        loss=criterion_outs.loss.detach().clone().cpu(),
        cw_loss=criterion_outs.cw_loss.detach().clone().cpu(),
        softmax_cw_loss=criterion_outs.softmax_cw_loss.detach().clone().cpu(),
        grad=_grad,
        target_class=criterion_outs.target_class.detach().clone().cpu(),
        logit=criterion_outs.logit.detach().clone().cpu(),
    )
    return criterion_outs_cpu


class TargetLabelCollecter:
    def __init__(self, n_examples) -> None:
        self.y_targets = [set() for _ in range(n_examples)]

    def update(self, indices, targets):
        for ind in torch.where(indices)[0]:
            self.y_targets[ind] |= set(targets[ind].tolist())

    def to_list(self):
        return list(map(lambda x: list(x), self.y_targets))
