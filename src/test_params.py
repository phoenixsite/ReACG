import gc
import os
import random
import sys

import numpy as np
import torch

from attacker import all_attacker
from core.criterion import Criterion, CriterionManager
from core.initial_point import (OutputDiversifiedSampling,
                                get_naive_initialpoint)
from evaluator_base import EvaluatorBase
from utils import (argparser, load_model_and_dataset, overwrite_config,
                   read_yaml, reproducibility, set_configurations,
                   setup_logger)
from utils.helper_functions import compute_accuracy

# import optuna
# import warnings
# from copy import deepcopy


class EvaluatorLinf(EvaluatorBase):
    def __init__(self, config, *args, **kwargs):
        super(EvaluatorLinf, self).__init__(config, *args, **kwargs)

    @torch.no_grad()
    def run(
        self,
        model,
        x_test,
        y_test,
        target_image_indices_all,
        target_indices,
        EXPERIMENT=False,
    ):

        param = self.config.param.copy()

        dataset = self.config.dataset
        # threat_model = self.config.threat_model

        device = (
            torch.device(f"cuda:{self.config.gpu_id}")
            if torch.cuda.is_available() and self.config.gpu_id is not None
            else torch.device("cpu")
        )
        self.attacker = all_attacker[self.config.threat_model][
            self.config.attacker_name
        ](
            epsilon=param.epsilon,
            device=device,
        )
        self.attacker.set_name(param)

        self.target_image_indices_all = target_image_indices_all.clone()
        self.target_indices = target_indices.clone()

        model_name, bs = self.config.model_name, self.config.batch_size

        model = model.to(device)
        model.eval()

        _criterion = Criterion(model)
        self.criterion = CriterionManager(_criterion)

        acc = torch.ones(
            (len(x_test),), dtype=bool
        )  # True iff the image is correctly classified.
        n_targets = self.config.n_targets
        # Remove images which is misclassified
        acc, cw_loss, y_targets = compute_accuracy(
            x_test, y_test, bs, model, device, K=n_targets
        )
        _clean_acc = (acc.sum() / acc.shape[0]) * 100
        if EXPERIMENT:
            acc = torch.ones((len(x_test),), dtype=bool)
        logger.info(f"clean acc: {_clean_acc:.2f}")

        n_backward = 0
        n_forward = len(x_test)
        n_classes = (
            10 if dataset == "cifar10" else 100 if dataset == "cifar100" else 1000
        )
        get_initialpoint = (
            OutputDiversifiedSampling(output_dim=n_classes)
            if param.initial_point == "odi"
            else get_naive_initialpoint
        )
        acc_whole = acc.clone()
        x_advs = x_test.clone()
        for n_restart in range(param.n_restarts):
            if EXPERIMENT:
                random.seed(n_restart)
                np.random.seed(n_restart)
                torch.manual_seed(n_restart)

            x_adv, search_information, n_forward, n_backward, accuracy = self.step(
                bs=bs,
                acc=acc,
                x_test=x_test,
                y_test=y_test,
                parameters=param,
                get_initialpoint=get_initialpoint,
                n_forward=n_forward,
                n_backward=n_backward,
                device=device,
            )

            adversarial_inds = torch.logical_and(acc, torch.logical_not(accuracy))
            x_advs[adversarial_inds] = x_adv[adversarial_inds].clone()
            acc_whole = torch.logical_and(acc_whole, accuracy)
            logger.info(f"ASR: {acc_whole.sum().item()} / {acc_whole.shape[0]}")
            acc = acc_whole.clone()

        _robust_acc, best_cw_loss, _ = compute_accuracy(x_advs, y_test, bs, model, device)
        return _robust_acc.sum().item(), best_cw_loss

@torch.no_grad()
def objective(a, b, c):
    torch.cuda.empty_cache()
    gc.collect()
    n_checkpoints = 0
    accum = 0
    w = int(a * config.param.max_iter)
    while accum + w < config.param.max_iter:
        accum += w
        n_checkpoints += 1
        w = max(w - int(b * config.param.max_iter), int(c * config.param.max_iter))
    if n_checkpoints < 4:
        return float("inf"), -float("inf")
    
    
    config_cp = config.copy()
    config_cp.param["a"] = a
    config_cp.param["b"] = b
    config_cp.param["c"] = c
    # config_cp.param["rho"] = rho
    evaluator = EvaluatorLinf(config_cp)
    ra, best_cw_loss = evaluator.run(
        model=model,
        x_test=x_test,
        y_test=y_test,
        target_image_indices_all=image_indices_all,
        target_indices=target_indices,
        EXPERIMENT=args.experiment,
    )
    return ra, best_cw_loss,


if __name__ == "__main__":
    BEST_CW=True
    is_best_cw = "MAXIMIZE_BEST_CW" if BEST_CW else "MINIMIZE_RA"
    # warnings.filterwarnings('ignore')
    sys.path.append("../src")
    reproducibility()
    parser = argparser()
    args = parser.parse_args()
    logger = setup_logger.setLevel(args.log_level)
    config = read_yaml(args.param)
    # overwrite by cmd_param
    if args.cmd_param is not None:
        config = overwrite_config(args.cmd_param, config)
    set_configurations(config, args)
    # config["param"]["initialpoint"]["dataset"] = config.dataset
    torch.set_num_threads(args.n_threads)
    os.environ["OMP_NUM_THREADS"] = str(args.n_threads)
    image_indices_yaml = args.image_indices
    with torch.no_grad():
        target_indices = torch.arange(0, config.n_examples, 1, dtype=int)
        image_indices_all = torch.arange(0, config.n_examples, 1, dtype=int)
        if image_indices_yaml is not None:
            # attack specified images
            target_indices = torch.tensor(read_yaml(image_indices_yaml).indices)
        model, x_test, y_test = load_model_and_dataset(
            config.model_name, config.dataset, config.n_examples, config.threat_model
        )
    try_these_parameters_first = [
        # dict(a=0.22, b=0.03, c=0.06, rho=0.75),
        dict(a=0.22, b=0.03, c=0.06),
    ]
    [
        dict(a=0.43, b=0.24, c=0.08), # best
        # dict(a=0.26, b=0.04, c=0.08), # Sehwag2021
        # dict(a=0.31, b=0.28, c=0.09), # Addepalli2021
        dict(a=0.42, b=0.25, c=0.10), # Addepalli2022
        # dict(a=0.69, b=0.69, c=0.05), # Ding2020
        
        # dict(a=0.35, b=0.19, c=0.10), # Sehwag, 10,000images, max cw (100trial), ASCG, 1.6231627464294434
        # dict(a=0.43, b=0.34, c=0.09), # Addepalli2022, 10,000images, max cw (300trial), ASCG, 0.6418462991714478
        # dict(a=0.24, b=0.21, c=0.09), # Addepalli2021, 10,000images, max cw (300trial), ASCG, 0.923687219619751
        # dict(a=0.61, b=0.50, c=0.08), # Rade_extra, 10,000images, max cw (100trial), ASCG, 1.2183432579040527
        # dict(a=0.54, b=0.40, c=0.06), # Gowal21_ddpm, 10,000images, max cw (100trial), ASCG, 2.243123769760132
        # dict(a=0.43, b=0.24, c=0.09), # Andriuschenko, 10,000images, max cw (100trial), ASCG, 1.1837222576141357
        # dict(a=0.48, b=0.20, c=0.08), # Wangbetter, 10,000images, max cw (100trial), ASCG, 1.1837222576141357
        # dict(a=0.34, b=0.15, c=0.08), # Sehwag, 1000images, max cw (300trial), ASCG, 1.6283122301101685
        # dict(a=0.32, b=0.07, c=0.06), # Addepalli2022, 1000images, max cw (300trial), ASCG, 0.6418462991714478
        # dict(a=0.20, b=0.01, c=0.05), # Addepalli2021, 1000images, max cw (300trial), ASCG, 0.923687219619751
        # dict(a=0.67, b=0.57, c=0.06), # Rade_extra, 1000images, max cw (100trial), ASCG, 1.2183432579040527
        # dict(a=0.52, b=0.31, c=0.08), # Rade_ddpm, 1000images, max cw (100trial), ASCG, 1.1788541078567505
        # dict(a=0.37, b=0.1, c=0.06), # Gowal21_ddpm, 1000images, max cw (100trial), ASCG, 2.243123769760132
        # dict(a=0.38, b=0.12, c=0.05), # Andriuschenko, 1000images, max cw (100trial), ASCG, 1.1837222576141357
        # dict(a=0.45, b=0.17, c=0.10), # Wangbetter, 1000images, max cw (100trial), ASCG, 1.1837222576141357
        # dict(a=0.30, b=0.07, c=0.06),
        # dict(a=0.40, b=0.15, c=0.06),
        # dict(a=0.41, b=0.15, c=0.05),
        # dict(a=0.53, b=0.25, c=0.06),
        # dict(a=0.33, b=0.01, c=0.1), # Sehwag, mifpe, ASCG, 100, A100
        # dict(a=0.37, b=0.13, c=0.1), # Sehwag, dlr, ASCG, 100, A100
        # dict(a=0.42, b=0.15, c=0.05), # Sehwag, dlr, ASCG, 1000, A100
        # dict(a=0.29, b=0.05, c=0.08), # Sehwag, cw, ASCG, 100, A100
        # dict(a=0.56, b=0.26, c=0.04), # Sehwag, cw, ASCG, 1000, A100
        # # dict(a=0.41, b=0.17, c=0.08), # Sehwag, dlr, ACG, 100, A100
        # # dict(a=0.35, b=0.12, c=0.07), # Sehwag, cw, ACG, 100, A100
        # dict(a=0.32, b=0.15, c=0.1), # Addepalli22, cw, ASCG, 100, A100
        # dict(a=0.28, b=0.04, c=0.08), # Addepalli22, dlr, ASCG, 100, A100
        # dict(a=0.30, b=0.07, c=0.09), # Addepalli22, cw, ASCG, 1000, A100
    ]
    print(config.model_name)
    for i, warm_start in enumerate(try_these_parameters_first):
        RA, loss = objective(**warm_start)
        print(i, RA, "{:.3f}".format(loss.mean(0).item()))
        torch.cuda.empty_cache()
        gc.collect()
