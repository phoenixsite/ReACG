import datetime
import os
import random
import sys
import time

import numpy as np
import torch
import yaml

from attacker import all_attacker
from core.criterion import Criterion, CriterionManager
from core.initial_point import (OutputDiversifiedSampling,
                                get_naive_initialpoint)
from evaluator_base import EvaluatorBase
from utils import (argparser, load_model_and_dataset, overwrite_config,
                   read_yaml, reproducibility, set_configurations,
                   setup_logger, tensor2csv)
from utils.helper_functions import compute_accuracy


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
        today = datetime.date.today().isoformat()
        _time = ":".join(datetime.datetime.now().time().isoformat().split(":")[:2])
        output_dir = os.path.join(self.config.output_dir, today, _time)
        os.makedirs(output_dir, exist_ok=True)

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
        stime = time.time()
        output_root_dir = os.path.join(
            output_dir,
            self.config.threat_model,
            dataset,
            model_name,
            self.attacker.name
        )
        os.makedirs(output_root_dir, exist_ok=True)

        _criterion = Criterion(model)
        self.criterion = CriterionManager(_criterion)

        acc = torch.ones(
            (len(x_test),), dtype=bool
        )  # True iff the image is correctly classified.
        # n_targets = 9 if dataset == "cifar10" else 13
        # n_targets = 9 if dataset == "cifar10" else 50 if dataset == "cifar100" else 100
        # n_targets = 9 if dataset == "cifar10" else 13 if dataset == "cifar100" else 20
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
            output_sub_dir = os.path.join(output_root_dir, str(n_restart))
            os.makedirs(output_sub_dir, exist_ok=True)
            for key in search_information:
                tensor2csv(
                    search_information[key], os.path.join(output_sub_dir, f"{key}.csv")
                )

            adversarial_inds = torch.logical_and(acc, torch.logical_not(accuracy))
            x_advs[adversarial_inds] = x_adv[adversarial_inds].clone()
            acc_whole = torch.logical_and(acc_whole, accuracy)
            logger.info(f"ASR: {acc_whole.sum().item()} / {acc_whole.shape[0]}")
            acc = acc_whole.clone()

        run_yaml_path = os.path.join(
            output_root_dir,
            "run.yaml",
        )
        if not os.path.exists(run_yaml_path):
            with open(run_yaml_path, "w") as file:
                yaml.dump(dict(self.config), file)

        # _robust_acc, _, _ = compute_accuracy(x_adv, y_test, bs, model, device)
        _robust_acc, _, _ = compute_accuracy(x_advs, y_test, bs, model, device)

        failed_indices_path = os.path.join(
            output_root_dir,
            "failed_indices.yaml",
        )
        if not os.path.exists(failed_indices_path):
            with open(failed_indices_path, "w") as file:
                yaml.dump({"indices": torch.where(_robust_acc)[0].tolist()}, file)

        robust_acc = 100 * (_robust_acc.sum() / self.config.n_examples)
        attack_success_rate = 100 - robust_acc
        n_forward += len(x_test)
        short_summary_path = os.path.join(output_root_dir, "short_summary.txt")
        msg = f"\ntotal time (sec) = {time.time() - stime:.3f}\nclean acc(%) = {_clean_acc:.2f}\nrobust acc(%) = {robust_acc:.2f}\nASR(%) = {attack_success_rate:.2f}\nForward = {n_forward}\nBackward = {n_backward}"
        with open(short_summary_path, "w") as f:
            f.write(msg)
        logger.info(msg)


@torch.no_grad()
def main(args):
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
    evaluator = EvaluatorLinf(config)
    evaluator.run(
        model=model,
        x_test=x_test,
        y_test=y_test,
        target_image_indices_all=image_indices_all,
        target_indices=target_indices,
        EXPERIMENT=args.experiment,
    )


if __name__ == "__main__":
    # torch.autograd.anomaly_mode.set_detect_anomaly(True)
    sys.path.append("../src")
    reproducibility()
    parser = argparser()
    args = parser.parse_args()
    logger = setup_logger.setLevel(args.log_level)
    main(args)
