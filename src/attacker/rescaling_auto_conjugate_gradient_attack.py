import gc
from typing import Callable, Dict, Union

import torch

from core.criterion import Criterion, CriterionManager
from utils import setup_logger
from utils.helper_functions import get_beta, normalize

from .base_attacker import BaseAttacker

logger = setup_logger(__name__)


class ReACG(BaseAttacker):
    """Rescaling Auto Conjugate Gradient (ReACG) Attack"""

    @classmethod
    def _set_name(cls, params):
        beta = params["beta"]
        max_iter = params["max_iter"]
        criterion_name = params["criterion_name"]
        use_cw_value = params["use_cw_value"]
        use_cw_flag = "UseCW" if use_cw_value else ""
        scaling_method = params["scaling_method"]
        use_linstep = params["use_linstep"]
        use_linstep_flag = "Linstep" if use_linstep else ""
        scaling_constant = str(params["scaling_constant"]) if scaling_method == "const" else ""
        return "-".join(
            [
                "ReACG",
                beta,
                criterion_name,
                str(max_iter),
                use_cw_flag,
                scaling_method,
                scaling_constant,
                use_linstep_flag
            ]
        )

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
        **kwargs,
    ):
        self.set_projection(x_nat)
        bs = x_nat.shape[0]
        max_iter = parameters["max_iter"]
        criterion_name = parameters["criterion_name"]
        initial_stepsize = parameters["initial_stepsize"]
        beta_method = parameters["beta"]
        use_clamp = parameters["use_clamp"]
        norm = parameters["normalization"]
        rho = parameters["rho"]
        move_to_best = parameters["move_to_best"]
        use_cw_value = parameters["use_cw_value"]
        scaling_method = parameters["scaling_method"]
        scaling_constant = parameters["scaling_constant"]
        use_linstep = parameters["use_linstep"]

        xk, begin, n_forward, n_backward = get_initialpoint(
            x_nat=x_nat,
            y_true=y_true,
            epsilon=self.epsilon,
            projection=self.projection,
            criterion=criterion,
            parameters=parameters,
            n_forward=n_forward,
            n_backward=n_backward,
        )
        criterion_outs = criterion(
            x=xk, y=y_true, criterion_name=criterion_name, enable_grad=True, *args, **kwargs
        )
        n_forward += bs
        n_backward += bs
        acc = criterion_outs.acc.cpu().clone()

        # initialize variables
        xk_1 = xk.clone()
        gradk = criterion_outs.grad.detach().clone()
        gradk_1 = torch.zeros_like(gradk)
        # gradk_1 = gradk.detach().clone()
        sk = None
        sk_1 = gradk.clone()
        betak = None
        loss_prev = criterion_outs.loss.cpu().clone()
        loss_best = loss_prev.clone()
        loss_best_cw = criterion_outs.cw_loss.cpu().clone()
        eta = torch.full((bs, 1, 1, 1), initial_stepsize, device=self.device)
        # parameters for APGD's stepsize update
        w = int(max_iter * parameters["a"])
        delta_w = int(max_iter * parameters["b"])
        min_delta_w = int(max_iter * parameters["c"])
        # w = int(max_iter * 0.55)
        # delta_w = int(max_iter * 0.2)
        # min_delta_w = int(max_iter * 0.06)
        # w = int(max_iter * 0.4)
        # delta_w = int(max_iter * 0.15)
        # min_delta_w = int(max_iter * 0.06)
        # w = int(max_iter * 0.22)
        # delta_w = int(max_iter * 0.03)
        # min_delta_w = int(max_iter * 0.06)
        checkpoint = w + begin
        # count_condition_1 = torch.zeros((bs,))
        reduced_last_check = torch.ones(size=(bs,), dtype=bool)
        best_loss_last_check = None

        x_adv = xk.clone().cpu()
        gradk_adv = gradk.clone().cpu()
        gradk_1_adv = gradk_1.clone().cpu()
        sk_1_adv = sk_1.clone().cpu()
        search_information = dict()
        search_information["current_loss"] = loss_prev.clone().unsqueeze(0)
        search_information["current_cw_loss"] = (
            criterion_outs.cw_loss.cpu().clone().unsqueeze(0)
        )
        search_information["best_loss"] = loss_best.clone().unsqueeze(0)
        search_information["best_cw_loss"] = loss_best_cw.clone().unsqueeze(0)
        search_information["eta"] = eta.cpu().squeeze().clone().unsqueeze(0)
        search_information["beta"] = torch.zeros((bs,)).unsqueeze(0)
        search_information["zk-xk_1"] = torch.zeros((bs,)).unsqueeze(0)
        search_information["xk-zk"] = torch.zeros((bs,)).unsqueeze(0)
        search_information["xk-xk_1"] = torch.zeros((bs,)).unsqueeze(0)
        search_information["descent_condition"] = torch.zeros((bs,)).unsqueeze(0)
        search_information["cosine"] = torch.zeros((bs,)).unsqueeze(0)
        search_information["target_class"] = criterion_outs.target_class.clone().unsqueeze(0)
        search_information["grad_norm"] = gradk.norm(p=2, dim=(1, 2, 3)).cpu().clone().unsqueeze(0)
        search_information["adjusted_beta"] = torch.zeros((bs,)).unsqueeze(0)
        search_information["avg_grad_ratio"] = torch.zeros((bs,)).unsqueeze(0)
        search_information["pow_avg_grad_ratio"] = torch.zeros((bs,)).unsqueeze(0)
        for i in range(begin, max_iter):
            # compute conjugate gradient
            if sk is None:
                sk = gradk.clone()
                betak = torch.zeros((bs,))
                normalize_inds = torch.zeros((bs, ))
                grad_ratio = torch.zeros((bs, ))
                grad_ratio_pow = torch.zeros((bs, ))
            else:
                betak = get_beta(
                    sk_1=sk_1,
                    gradk_1=gradk_1,
                    gradk=gradk,
                    method=beta_method,
                    use_clamp=use_clamp,
                )
                if scaling_method == "const":
                    # const = pow(10, 5)
                    betak_norm = get_beta(
                        sk_1=sk_1,
                        gradk_1=gradk_1 / scaling_constant,
                        gradk=gradk / scaling_constant,
                        method=beta_method,
                        use_clamp=use_clamp,
                    )
                elif scaling_method == "norm":
                    betak_norm = get_beta(
                        sk_1=sk_1,
                        gradk_1=normalize(gradk_1, norm="l2"),
                        gradk=normalize(gradk, norm="l2"),
                        method=beta_method,
                        use_clamp=use_clamp,
                    )
                elif scaling_method == "zero":
                    betak_norm = torch.zeros_like(betak)
                else:
                    raise NotImplementedError(f"scaling method [{scaling_method}] is not implemented.")
                grad_ratio_value = (gradk / sk_1).abs().reshape((bs, -1))
                # grad_ratio = grad_ratio_value.max(1).values
                grad_ratio = (
                    grad_ratio_value.mean(1)
                    # grad_ratio_value.quantile(q=0.3, dim=1, keepdim=False)
                    # grad_ratio_value.cpu().median(1)[0].to(self.device)
                )  # betaがこれよりも大きい→sk_1の値が支配的になる
                grad_ratio_pow = grad_ratio_value.pow(2).mean(1)
                normalize_inds = betak.abs().reshape((bs,)) > grad_ratio  # この条件次第で調節する。
                # normalize_inds = betak.abs().reshape((bs,)) > grad_ratio * 5/4  # この条件次第で調節する。
                smaller_beta_inds = (betak.abs() > betak_norm.abs()).reshape((bs,))
                normalize_inds = torch.logical_and(normalize_inds, smaller_beta_inds)
                if normalize_inds.any():
                    logger.warning(f"beta adjusted: #{normalize_inds.sum().item():d}")
                betak[normalize_inds] = betak_norm[normalize_inds].clone()
                # N = 6
                # if len( search_information["best_cw_loss"]) > N:
                #     restarting_idxs = torch.logical_and((eta < self.epsilon).reshape((bs,)), (search_information["best_cw_loss"][-1] - search_information["best_cw_loss"][-N:-1].mean(0) < 1e-3).to(self.device))
                #     logger.warning(f"#restart: {restarting_idxs.sum().item()}")
                #     betak[restarting_idxs] = 0.0
                #     eta[restarting_idxs] = initial_stepsize
                sk = gradk + betak * sk_1

            logger.debug(f"beta({beta_method}) = {betak.mean().item():.4f}")
            if use_linstep:
                eta[:] = initial_stepsize * (1.0 - i / max_iter)
            # move to conjugate gradient direction
            zk = xk + eta * normalize(sk, norm)
            xk = self.projection(zk)
            # compute objective value
            criterion_outs = criterion(
                x=xk, y=y_true, criterion_name=criterion_name, enable_grad=True, *args, **kwargs
            )
            n_forward += bs
            n_backward += bs
            loss_current = criterion_outs.loss.cpu().clone()
            loss_current_cw = criterion_outs.cw_loss.cpu().clone()

            expected_movement = (zk - xk_1).norm(p=2, dim=(1, 2, 3)).cpu()
            drawback_by_projection = (xk - zk).norm(p=2, dim=(1, 2, 3)).cpu()
            actual_movement = (xk - xk_1).norm(p=2, dim=(1, 2, 3)).cpu()
            descent_condition = (
                (-gradk * normalize(sk, norm)).sum(dim=(1, 2, 3))
                / gradk.norm(p=2, dim=(1, 2, 3)).pow(2)
            ).cpu()
            cosine = (
                (-gradk * normalize(sk, norm)).sum(dim=(1, 2, 3))
                / gradk.norm(p=2, dim=(1, 2, 3))
                / normalize(sk, norm).norm(p=2, dim=(1, 2, 3))
            ).cpu()

            # increment count for stepsize condition 1
            # increment_counter_inds = loss_prev < loss_current
            # count_condition_1[increment_counter_inds] += 1
            loss_prev = loss_current.clone()

            # update variables for next iteration
            xk_1 = xk.detach().clone()
            gradk_1 = gradk.detach().clone()
            sk_1 = sk.detach().clone()
            gradk = criterion_outs.grad.detach().clone()

            # update solution
            improved_inds_cw = torch.logical_or(
                loss_best_cw < loss_current_cw,
                torch.logical_and(acc, torch.logical_not(criterion_outs.acc.cpu())),
            )
            loss_best_cw[improved_inds_cw] = loss_current_cw[improved_inds_cw].clone()
            improved_inds_org = torch.logical_or(
                loss_best < loss_current,
                torch.logical_and(acc, torch.logical_not(criterion_outs.acc.cpu())),
            )
            loss_best[improved_inds_org] = loss_current[improved_inds_org].clone()
            if use_cw_value:
                improved_inds = improved_inds_cw.clone()
            else:
                improved_inds = improved_inds_org.clone()
            x_adv[improved_inds] = xk[improved_inds].cpu().clone()
            gradk_adv[improved_inds] = gradk[improved_inds].cpu().clone()
            gradk_1_adv[improved_inds] = gradk_1[improved_inds].cpu().clone()
            sk_1_adv[improved_inds] = sk_1[improved_inds].cpu().clone()
            assert (xk[improved_inds].cpu() == x_adv[improved_inds]).all()
            # print((gradk_adv - gradk_1_adv == 0).all(3).all(2).all(1).sum()) # 初期点がbestの場合、gradk == gradk_1になる場合がある

            acc = torch.logical_and(acc, criterion_outs.acc.cpu())

            # update search information
            search_information["current_loss"] = torch.cat(
                [search_information["current_loss"], loss_current.clone().unsqueeze(0)]
            )
            search_information["current_cw_loss"] = torch.cat(
                [
                    search_information["current_cw_loss"],
                    criterion_outs.cw_loss.cpu().clone().unsqueeze(0),
                ]
            )
            search_information["best_loss"] = torch.cat(
                [search_information["best_loss"], loss_best.clone().unsqueeze(0)]
            )
            search_information["best_cw_loss"] = torch.cat(
                [search_information["best_cw_loss"], loss_best_cw.clone().unsqueeze(0)]
            )
            search_information["eta"] = torch.cat(
                [search_information["eta"], eta.squeeze().cpu().clone().unsqueeze(0)]
            )
            search_information["beta"] = torch.cat(
                [
                    search_information["beta"],
                    betak.reshape((bs,)).cpu().clone().unsqueeze(0),
                ]
            )
            search_information["zk-xk_1"] = torch.cat(
                [search_information["zk-xk_1"], expected_movement.clone().unsqueeze(0)]
            )
            search_information["xk-zk"] = torch.cat(
                [
                    search_information["xk-zk"],
                    drawback_by_projection.clone().unsqueeze(0),
                ]
            )
            search_information["xk-xk_1"] = torch.cat(
                [search_information["xk-xk_1"], actual_movement.clone().unsqueeze(0)]
            )
            search_information["descent_condition"] = torch.cat(
                [
                    search_information["descent_condition"],
                    descent_condition.clone().unsqueeze(0),
                ]
            )
            search_information["cosine"] = torch.cat(
                [search_information["cosine"], cosine.clone().unsqueeze(0)]
            )
            search_information["target_class"] = torch.cat(
                [
                    search_information["target_class"],
                    criterion_outs.target_class.clone().unsqueeze(0),
                ]
            )
            search_information["grad_norm"] = torch.cat(
                [
                    search_information["grad_norm"],
                    gradk.norm(p=2, dim=(1, 2, 3)).cpu().clone().unsqueeze(0),
                ]
            )
            search_information["adjusted_beta"] = torch.cat(
                [
                    search_information["adjusted_beta"],
                    normalize_inds.cpu().clone().unsqueeze(0),
                ]
            )
            search_information["avg_grad_ratio"] = torch.cat(
                [
                    search_information["avg_grad_ratio"],
                    grad_ratio.cpu().clone().unsqueeze(0),
                ]
            )
            search_information["pow_avg_grad_ratio"] = torch.cat(
                [
                    search_information["pow_avg_grad_ratio"],
                    grad_ratio_pow.cpu().clone().unsqueeze(0),
                ]
            )

            if (not use_linstep) and (i + 1 == checkpoint):
                if best_loss_last_check is None:
                    best_loss_last_check = loss_best.clone()

                def check_oscillation(iteration, loss_steps):
                    num_updates = torch.zeros(loss_steps.shape[1])
                    for counter5 in range(w):
                        if iteration - counter5 - 1 < 0:
                            num_updates += (
                                loss_steps[1:, :][iteration - counter5, :] > 0
                            )
                        else:
                            num_updates += (
                                loss_steps[1:, :][iteration - counter5, :]
                                > loss_steps[1:, :][iteration - counter5 - 1, :]
                            )
                    return num_updates <= rho * w

                condition_1 = check_oscillation(
                    i - begin, search_information["current_loss"]
                )
                # condition_1 = count_condition_1 < rho * w
                condition_2 = torch.logical_and(
                    best_loss_last_check >= loss_best,
                    torch.logical_not(reduced_last_check),
                )
                condition = torch.logical_or(condition_1, condition_2).to(self.device)
                eta[condition] /= 2
                if move_to_best:
                    xk[condition] = x_adv.to(self.device)[condition].clone()
                    gradk[condition] = gradk_adv.to(self.device)[condition].clone()
                    gradk_1[condition] = gradk_1_adv.to(self.device)[condition].clone()
                    sk_1[condition] = sk_1_adv.to(self.device)[condition].clone()
                    assert (xk[condition] == x_adv.to(self.device)[condition]).all()
                # count_condition_1[:] = 0
                w = max(w - delta_w, min_delta_w)
                checkpoint += w
                reduced_last_check = condition.cpu().clone()
                best_loss_last_check = loss_best.clone()

            log_msg = (
                f"iteration {i}:",
                # f"{loss_current.sum().item():.1f}",
                # f"{loss_best.sum().item():.1f}",
                f"{search_information['current_cw_loss'][-1].sum().item():.3f}",
                f"{search_information['current_cw_loss'].max(0).values.sum().item():.3f}",
                f"{eta.mean().item():.3f}",
                # f"{expected_movement.mean().item():.3f}",
                # f"{drawback_by_projection.mean().item():.3f}",
                # f"{actual_movement.mean().item():.3f}",
                f"{actual_movement.mean().item() / (expected_movement.mean().item() + 1e-12) :.3f}",
                f"{acc.sum().item() / acc.shape[0] * 100:.2f}",
            )
            logger.info(" ".join(log_msg))
            if i % 5 == 0:
                gc.collect()

        self.check_feasibility(x_adv)
        return (
            x_adv.cpu(),
            gradk_adv.cpu(),
            search_information,
            n_forward,
            n_backward,
            acc,
        )