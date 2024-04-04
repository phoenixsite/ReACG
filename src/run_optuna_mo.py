import os
import random
import sys

import numpy as np
import torch

from attacker import all_attacker
from core.criterion import Criterion, CriterionManager
from core.initial_point import (OutputDiversifiedSampling,
                                PredictionAwareSampling,
                                get_naive_initialpoint)
from evaluator_base import EvaluatorBase
from utils import (argparser, load_model_and_dataset, overwrite_config,
                   read_yaml, reproducibility, set_configurations,
                   setup_logger)
from utils.helper_functions import compute_accuracy

import optuna
import warnings
from copy import deepcopy


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
            else PredictionAwareSampling(output_dim=n_classes)
            if param.initial_point == "pas"
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
        atk_succ_mask = torch.logical_not(_robust_acc.to(bool))
        return _robust_acc.sum().item(), best_cw_loss, atk_succ_mask

@torch.no_grad()
def objective(trial):
    a = trial.suggest_discrete_uniform(
        "a",
        0.02, 0.9, 0.01
        # log=True,
    )
    b = trial.suggest_discrete_uniform(
        "b",
        0.01, a, 0.01,
        # log=True,
    )
    c = trial.suggest_discrete_uniform(
        "c",
        0.01, 0.1, 0.01
        # log=True,
    )
    # rho = trial.suggest_discrete_uniform(
    #     "rho",
    #     0.0, 1.0, 0.05
    #     # log=True,
    # )
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
    ra, best_cw_loss, atk_succ_mask = evaluator.run(
        model=model,
        x_test=x_test,
        y_test=y_test,
        target_image_indices_all=image_indices_all,
        target_indices=target_indices,
        EXPERIMENT=args.experiment,
    )
    return ra, best_cw_loss.mean().item()


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
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
    target_indices = torch.arange(0, config.n_examples, 1, dtype=int)
    image_indices_all = torch.arange(0, config.n_examples, 1, dtype=int)
    if image_indices_yaml is not None:
        # attack specified images
        target_indices = torch.tensor(read_yaml(image_indices_yaml).indices)
    model, x_test, y_test = load_model_and_dataset(
        config.model_name, config.dataset, config.n_examples, config.threat_model
    )
    import logging
    study_name = "-".join(["hypera-opt", config.attacker_name, config.model_name, config.param.criterion_name, str(config.param.max_iter), str(config.n_examples), "MULTI_OBJECTIVE"])
    optuna.logging.get_logger("optuna").addHandler(logging.FileHandler(f'./{study_name}.log'))
    study = optuna.create_study(
        study_name=study_name,
        storage='sqlite:///../optuna_study_mo.db',
        load_if_exists=True,
        directions=["minimize", "maximize"],
        # sampler=optuna.multi_objective.samplers.NSGAIIMultiObjectiveSampler()
    )
    # fig = optuna.visualization.plot_param_importances(study)
    # fig = optuna.visualization.plot_contour(study)
    # fig.write_image("../test.png")
    # exit()
    try_these_parameters_first = [
        dict(a=0.22, b=0.03, c=0.06),
        dict(a=0.43, b=0.24, c=0.08), 
    ]
    for warm_start in try_these_parameters_first:
        study.enqueue_trial(warm_start)
    study.optimize(objective, n_trials=200)
    # print("best : params / value =  " , study.best_params, " / ", study.best_value)
    import matplotlib.pyplot as plt
    # 最適化過程で得た履歴データの取得。get_trials()メソッドを使用
    trials = {str(trial.values): trial for trial in study.get_trials()}
    trials = list(trials.values())
    # グラフにプロットするため、目的変数をリストに格納する
    y1_all_list = []
    y2_all_list = []
    for i, trial in enumerate(trials, start=1):
        if trial.values is None:
            continue
        y1_all_list.append(trial.values[0])
        y2_all_list.append(trial.values[1])

    # パレート解の取得。get_pareto_front_trials()メソッドを使用
    # trials = {str(trial.values): trial for trial in study.get_pareto_front_trials()}
    # trials = list(trials.values())
    # trials.sort(key=lambda t: t.values)
    # # グラフプロット用にリストで取得。またパレート解の目的変数と説明変数をcsvに保存する
    # y1_list = []
    # y2_list = []
    # with open(f'{study_name}_pareto_data_real.csv', 'w') as f:
    #     for i, trial in enumerate(trials, start=1):
    #         if i == 1:
    #             columns_name_str = 'trial_no,y1,y2'
    #         data_list = []
    #         data_list.append(trial.number)
    #         y1_value = trial.values[0]
    #         y2_value = trial.values[1]
    #         y1_list.append(y1_value)
    #         y2_list.append(y2_value)
    #         data_list.append(y1_value)
    #         data_list.append(y2_value)    
    #         for key, value in trial.params.items():
    #             data_list.append(value)
    #             if i == 1:
    #                 columns_name_str += ',' + key 
    #         if i == 1:
    #             f.write(columns_name_str + '\n')
    #         data_list = list(map(str, data_list))
    #         data_list_str = ','.join(data_list)
    #         f.write(data_list_str + '\n')

    # パレート解を図示
    plt.rcParams["font.size"] = 16
    plt.figure(dpi=120)
    plt.title("multiobjective optimization")
    plt.xlabel("Y1")
    plt.ylabel("Y2")
    plt.grid()
    plt.scatter(y1_all_list, y2_all_list, c='blue', label='all trials')
    # plt.scatter(y1_list, y2_list, c='red', label='pareto front')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{study_name}.png")
    plt.close()
