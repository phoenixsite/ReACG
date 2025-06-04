import math
from typing import Callable, Dict, Union

import torch

from core.criterion import Criterion, CriterionManager
from utils import setup_logger

logger = setup_logger(__name__)


class EvaluatorBase:
    """Management of the input images and output of the attack

    Attributes
    ----------
    config : dict
        Config of the evaluation
    """

    def __init__(self, config):
        self.config = config
        self.criterion: Union[Criterion, CriterionManager] = None

    @torch.no_grad()
    def step(
        self,
        bs: int,
        acc: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        parameters: Dict,
        get_initialpoint: Callable,
        n_forward: int,
        n_backward: int,
        device: Union[torch.device, str],
        y_target: torch.Tensor =None,
        *args,
        **kwargs,
    ):
        """Conduct an adversarial attack"""
        logger.info("[ Step ]")
        search_information = dict()
        x, y = x_test, y_test
        n_examples = acc[self.target_indices].sum().item()
        nbatches = math.ceil(n_examples / bs)
        n_success = 0
        logger.info(f"idx {-1}: ASR = {n_success} / 0")
        _accuracy = acc.clone()
        x_adv = x.clone()

        for idx in range(nbatches):
            begin = idx * bs
            end = min((idx + 1) * bs, n_examples)

            target_image_indices = self.target_image_indices_all[self.target_indices][
                acc[self.target_indices]
            ][begin:end]

            if len(x[target_image_indices]) > 0:
                if y_target is not None:
                    _y_target = y_target[target_image_indices].clone().to(device)
                else:
                    _y_target = None
                (
                    x_adv_batch,
                    gradk_adv_batch,
                    search_information_batch,
                    n_forward,
                    n_backward,
                    accuracy,
                ) = self.attacker.attack(
                    x_nat=x[target_image_indices].clone().to(device),
                    y_true=y[target_image_indices].clone().to(device),
                    y_target=_y_target,
                    parameters=parameters,
                    criterion=self.criterion,
                    get_initialpoint=get_initialpoint,
                    n_forward=n_forward,
                    n_backward=n_backward,
                    *args,
                    **kwargs,
                )
                if len(search_information) == 0:
                    for key in search_information_batch:
                        # [iteration, batch_size]
                        shape = search_information_batch[key].shape 
                        search_information[key] = -torch.ones((len(x_test), shape[0]))
                for key in search_information:
                    tmp = search_information[key].clone()
                    if (
                        end - begin == 1
                        and len(search_information_batch[key].shape) == 1
                    ):
                        search_information_batch[key] = search_information_batch[
                            key
                        ].unsqueeze(1)
                    tmp[target_image_indices] = (
                        search_information_batch[key].permute(1, 0).clone()
                    )
                    search_information[key] = tmp.clone()
                x_adv[target_image_indices] = x_adv_batch.clone()

                _accuracy[target_image_indices] = accuracy.clone().cpu()
                n_success += torch.logical_not(accuracy).sum().item()
                logger.info(f"idx {idx}: ASR = {n_success} / {end}")
            else:
                logger.warning(f"#target image = 0 at batch {idx}.")

        return x_adv, search_information, n_forward, n_backward, _accuracy

    def run(
        self,
        model,
        x_test,
        y_test,
        target_image_indices_all,
        target_indices,
        EXPORT_LEVEL=60,
        EXPERIMENT=False,
        save_adversarial=False,
    ):
        raise NotImplementedError
