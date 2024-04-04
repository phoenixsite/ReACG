import gc
from typing import Callable, Dict, Union

import torch

from core.criterion import Criterion, CriterionManager
from utils import setup_logger
from utils.helper_functions import normalize

from .base_attacker import BaseAttacker

logger = setup_logger(__name__)


class APGD(BaseAttacker):
    """Auto Projected Gradient Descent (APGD) Attack"""

    def __init__(self, *args, **kwargs) -> None:
        super(APGD, self).__init__(*args, **kwargs)

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
        norm = parameters["normalization"]
        rho = parameters["rho"]
        momentum_alpha = parameters["momentum_alpha"]
        move_to_best = parameters["move_to_best"]
        use_cw_value = parameters["use_cw_value"]

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
        # gradk_1 = gradk.detach().clone()
        loss_prev = criterion_outs.loss.cpu().clone()
        loss_best = loss_prev.clone()
        loss_best_cw = criterion_outs.cw_loss.cpu().clone()
        eta = torch.full((bs, 1, 1, 1), initial_stepsize, device=self.device)
        # parameters for APGD's stepsize update
        w = int(max_iter * 0.22)
        delta_w = int(max_iter * 0.03)
        min_delta_w = int(max_iter * 0.06)
        checkpoint = w + begin
        # count_condition_1 = torch.zeros((bs,))
        reduced_last_check = torch.ones(size=(bs,), dtype=bool)
        best_loss_last_check = None

        x_adv = xk.clone().cpu()
        x_1_adv = xk_1.clone().cpu()
        gradk_adv = gradk.clone().cpu()
        search_information = dict()
        search_information["current_loss"] = loss_prev.clone().unsqueeze(0)
        search_information["current_cw_loss"] = (
            criterion_outs.cw_loss.cpu().clone().unsqueeze(0)
        )
        search_information["best_loss"] = loss_best.clone().unsqueeze(0)
        search_information["best_cw_loss"] = loss_best_cw.clone().unsqueeze(0)
        search_information["eta"] = eta.cpu().squeeze().clone().unsqueeze(0)
        search_information["zk-xk_1"] = torch.zeros((bs,)).unsqueeze(0)
        search_information["xk-zk"] = torch.zeros((bs,)).unsqueeze(0)
        search_information["xk-xk_1"] = torch.zeros((bs,)).unsqueeze(0)
        search_information["descent_condition"] = torch.zeros((bs,)).unsqueeze(0)
        search_information["cosine"] = torch.zeros((bs,)).unsqueeze(0)
        search_information["target_class"] = criterion_outs.target_class.clone().unsqueeze(0)
        search_information["grad_norm"] = gradk.norm(p=2, dim=(1, 2, 3)).cpu().clone().unsqueeze(0)
        for i in range(begin, max_iter):
            # move to gradient direction
            _xk = xk.detach().clone()
            grad2 = xk.detach() - xk_1
            zk = xk + eta * normalize(gradk, norm)
            xk = self.projection(zk)

            # compute moving distance
            expected_movement = (zk - xk_1).norm(p=2, dim=(1, 2, 3)).cpu()
            drawback_by_projection = (xk - zk).norm(p=2, dim=(1, 2, 3)).cpu()
            actual_movement = (xk - xk_1).norm(p=2, dim=(1, 2, 3)).cpu()
            descent_condition = (
                (-gradk * normalize(gradk, norm)).sum(dim=(1, 2, 3))
                / gradk.norm(p=2, dim=(1, 2, 3)).pow(2)
            ).cpu()
            cosine = (
                (-gradk * normalize(gradk, norm)).sum(dim=(1, 2, 3))
                / gradk.norm(p=2, dim=(1, 2, 3))
                / normalize(gradk, norm).norm(p=2, dim=(1, 2, 3))
            ).cpu()
            # move to momentum direction
            a = momentum_alpha if i > begin else 1.0
            xk = self.projection(_xk + a * (xk - _xk) + (1.0 - a) * grad2)
            # compute objective value
            criterion_outs = criterion(
                x=xk, y=y_true, criterion_name=criterion_name, enable_grad=True, *args, **kwargs
            )
            n_forward += bs
            n_backward += bs
            loss_current = criterion_outs.loss.cpu().clone()
            loss_current_cw = criterion_outs.cw_loss.cpu().clone()

            # increment count for stepsize condition 1
            # increment_counter_inds = loss_prev < loss_current
            # count_condition_1[increment_counter_inds] += 1
            loss_prev = loss_current.clone()

            # update variables for next iteration
            xk_1 = _xk.detach().clone()
            # gradk_1 = gradk.detach().clone()
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
            x_1_adv[improved_inds] = xk_1[improved_inds].cpu().clone()
            gradk_adv[improved_inds] = gradk[improved_inds].cpu().clone()
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

            if i + 1 == checkpoint:
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
                    return num_updates <= rho * w  # * torch.ones_like(num_updates)

                condition_1 = check_oscillation(
                    i - begin, search_information["current_loss"]
                )
                # count_condition_1 = check_oscillation(i - begin, search_information["current_loss"])
                # condition_1 = count_condition_1 < rho * w
                condition_2 = torch.logical_and(
                    best_loss_last_check >= loss_best,
                    torch.logical_not(reduced_last_check),
                )
                condition = torch.logical_or(condition_1, condition_2).to(self.device)
                eta[condition] /= 2
                if move_to_best:
                    xk[condition] = x_adv.to(self.device)[condition].clone()
                    # xk_1[condition] = x_1_adv.to(self.device)[condition].clone()
                    gradk[condition] = gradk_adv.to(self.device)[condition].clone()
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

    def set_name(self, parameters: Dict):
        beta = "*"
        max_iter = parameters["max_iter"]
        criterion_name = parameters["criterion_name"]
        use_cw_value = parameters["use_cw_value"]
        use_cw_flag = "UseCW" if use_cw_value else ""
        self.name = "-".join(
            [
                "APGD",
                beta,
                criterion_name,
                str(max_iter),
                use_cw_flag,
            ]
        )


# import numpy as np
# class APGDWOModule(BaseAttacker):
#    """Auto Projected Gradient Descent (APGD) Attack"""

# def __init__(self, *args, **kwargs) -> None:
#     super(APGDWOModule, self).__init__(*args, **kwargs)

# def attack(
#     self,
#     x_nat: torch.Tensor,
#     y_true: torch.Tensor,
#     parameters: Dict,
#     criterion: Union[Criterion, CriterionManager],
#     get_initialpoint: Callable,
#     n_forward: int,
#     n_backward: int,
#     *args,
#     **kwargs,
# ):

#     bs = x_nat.shape[0]
#     self.max_iter = parameters["max_iter"]
#     criterion_name = parameters["criterion_name"]
#     initial_stepsize = parameters["initial_stepsize"]
#     rho = parameters["rho"]
#     self.momentum_alpha = parameters["momentum_alpha"]
#     self.thr_decr = rho
#     self.eot_iter = 1
#     x = x_nat.clone()
#     self.model = criterion.criterion.model
#     self.criterion = criterion.criterion.criterions[criterion_name]
#     y = y_true.to(self.device)

#     self.max_iter_2, self.max_iter_min, self.size_decr = (
#         max(int(0.22 * self.max_iter), 1),
#         max(int(0.06 * self.max_iter), 1),
#         max(int(0.03 * self.max_iter), 1),
#     )
#     message = f"parameters: {self.max_iter} {self.max_iter_2} {self.max_iter_min} {self.size_decr}"
#     logger.debug(message)

#     x_nat = x_nat.to(self.device)
#     self.set_projection(x_nat)
#     # iteration.
#     logits = self.model(x_nat.to(self.device))

#     x_adv, begin, n_forward, n_backward = get_initialpoint(
#         x_nat=x_nat,
#         y_true=y_true,
#         epsilon=self.epsilon,
#         projection=self.projection,
#         criterion=criterion,
#         parameters=parameters,
#         n_forward=n_forward,
#         n_backward=n_backward,
#     )

#     x_adv = x_adv.to(self.device)

#     assert (x_adv >= 0).all().item()
#     assert (x_adv <= 1).all().item()

#     x_best = x_adv.clone()
#     x_best_adv = x_adv.clone()
#     loss_steps = torch.zeros([self.max_iter, x.shape[0]])

#     x_adv.requires_grad_()
#     grad = torch.zeros_like(x_adv)
#     for _ in range(self.eot_iter):
#         with torch.enable_grad():
#             logits = self.model(x_adv)  # 1 forward pass (eot_iter = 1)
#             loss_indiv = self.criterion(logits, y)
#             loss = loss_indiv.sum()

#         # 1 backward pass (eot_iter = 1)
#         grad += torch.autograd.grad(loss, [x_adv])[0].detach()

#     grad /= float(self.eot_iter)
#     grad_best = grad.clone()

#     loss_best = loss_indiv.detach().clone()

#     step_size = (
#         self.epsilon
#         * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach()
#         * torch.Tensor([2.0]).to(self.device).detach().reshape([1, 1, 1, 1])
#     )  ##

#     alpha = 2.0

#     step_size = (
#         alpha
#         * self.epsilon
#         * torch.ones([x_nat.shape[0], *([1] * len(x_nat.shape[1:]))])
#         .to(self.device)
#         .detach()
#     )

#     x_adv_old = x_adv.clone()
#     counter = 0
#     k = self.max_iter_2 + 0

#     n_fts = x_nat.shape[-3] * x_nat.shape[-2] * x_nat.shape[-1]
#     u = torch.arange(x_nat.shape[0], device=self.device)

#     counter3 = 0

#     loss_best_last_check = loss_best.clone()
#     reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
#     n_reduced = 0

#     for iteration in range(self.max_iter):
#         # gradient step
#         with torch.no_grad():
#             x_adv = x_adv.detach()
#             grad2 = x_adv - x_adv_old
#             x_adv_old = x_adv.clone()

#             a = self.momentum_alpha if iteration >= 1 else 1.0

#             # if self.norm == "Linf":
#             x_adv_1 = x_adv + step_size * torch.sign(grad)

#             x_adv_1 = torch.clamp(
#                 torch.min(
#                     torch.max(x_adv_1, x_nat - self.epsilon),
#                     x_nat + self.epsilon,
#                 ),
#                 0.0,
#                 1.0,
#             )

#             x_adv_1 = torch.clamp(
#                 torch.min(
#                     torch.max(
#                         x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
#                         x_nat - self.epsilon,
#                     ),
#                     x_nat + self.epsilon,
#                 ),
#                 0.0,
#                 1.0,
#             )

#             # elif self.norm == "L2":
#             #     x_adv_1 = x_adv + step_size * grad / (
#             #         (grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12
#             #     )
#             #     x_adv_1 = torch.clamp(
#             #         x
#             #         + (x_adv_1 - x_nat)
#             #         / (
#             #             ((x_adv_1 - x_nat) ** 2)
#             #             .sum(dim=(1, 2, 3), keepdim=True)
#             #             .sqrt()
#             #             + 1e-12
#             #         )
#             #         * torch.min(
#             #             self.epsilon
#             #             * torch.ones(x_nat.shape).to(self.device).detach(),
#             #             ((x_adv_1 - x_nat) ** 2)
#             #             .sum(dim=(1, 2, 3), keepdim=True)
#             #             .sqrt(),
#             #         ),
#             #         0.0,
#             #         1.0,
#             #     )
#             #     x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
#             #     x_adv_1 = torch.clamp(
#             #         x_nat
#             #         + (x_adv_1 - x_nat)
#             #         / (
#             #             ((x_adv_1 - x_nat) ** 2)
#             #             .sum(dim=(1, 2, 3), keepdim=True)
#             #             .sqrt()
#             #             + 1e-12
#             #         )
#             #         * torch.min(
#             #             self.epsilon
#             #             * torch.ones(x_nat.shape).to(self.device).detach(),
#             #             ((x_adv_1 - x_nat) ** 2)
#             #             .sum(dim=(1, 2, 3), keepdim=True)
#             #             .sqrt()
#             #             + 1e-12,
#             #         ),
#             #         0.0,
#             #         1.0,
#             #     )

#             # elif self.norm == "L1":
#             #     grad_topk = grad.abs().view(x_nat.shape[0], -1).sort(-1)[0]
#             #     topk_curr = torch.clamp(
#             #         (1.0 - topk) * n_fts, min=0, max=n_fts - 1
#             #     ).long()
#             #     grad_topk = grad_topk[u, topk_curr].view(
#             #         -1, *[1] * (len(x_nat.shape) - 1)
#             #     )
#             #     sparsegrad = grad * (grad.abs() >= grad_topk).float()
#             #     x_adv_1 = x_adv + step_size * sparsegrad.sign() / (
#             #         sparsegrad.sign()
#             #         .abs()
#             #         .view(x.shape[0], -1)
#             #         .sum(dim=-1)
#             #         .view(-1, *[1] * (len(x_nat.shape) - 1))
#             #         + 1e-10
#             #     )

#             #     delta_u = x_adv_1 - x_nat
#             #     delta_p = L1_projection(x_nat, delta_u, self.epsilon)
#             #     x_adv_1 = x_nat + delta_u + delta_p

#             x_adv = x_adv_1 + 0.0

#         # get gradient
#         x_adv.requires_grad_()
#         grad = torch.zeros_like(x_adv)
#         for _ in range(self.eot_iter):
#             with torch.enable_grad():
#                 logits = self.model(x_adv)  # 1 forward pass (eot_iter = 1)
#                 loss_indiv = self.criterion(logits, y)
#                 loss = loss_indiv.sum()
#             # 1 backward pass (eot_iter = 1)
#             grad += torch.autograd.grad(loss, [x_adv])[0].detach()
#         grad /= float(self.eot_iter)

#         pred = logits.detach().max(1)[1] == y

#         x_best_adv[(pred == 0).nonzero().squeeze()] = (
#             x_adv[(pred == 0).nonzero().squeeze()] + 0.0
#         )

#         message = "\n iteration: {} - Loss: {:.6f} - Best loss: {:.6f}".format(
#             iteration, loss.sum(), loss_best.sum()
#         )
#         logger.info(message)

#         # check step size
#         with torch.no_grad():
#             y1 = loss_indiv.detach().clone()
#             loss_steps[iteration] = y1.cpu() + 0
#             ind = (y1 > loss_best).nonzero().squeeze()
#             x_best[ind] = x_adv[ind].clone()
#             grad_best[ind] = grad[ind].clone()
#             loss_best[ind] = y1[ind] + 0

#             counter3 += 1

#             if counter3 == k:
#                 # if self.norm in {"Linf", "L2"}:
#                 fl_oscillation = self.check_oscillation(
#                     loss_steps.detach().cpu().numpy(),
#                     iteration,
#                     k,
#                     loss_best.detach().cpu().numpy(),
#                     k3=self.thr_decr,
#                 )
#                 fl_reduce_no_impr = (~reduced_last_check) * (
#                     loss_best_last_check.cpu().numpy()
#                     >= loss_best.cpu().numpy()
#                 )
#                 fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
#                 reduced_last_check = np.copy(fl_oscillation)
#                 loss_best_last_check = loss_best.clone()

#                 ## stepsize.
#                 if np.sum(fl_oscillation) > 0:
#                     step_size[u[fl_oscillation]] /= 2.0
#                     n_reduced = fl_oscillation.astype(float).sum()

#                     fl_oscillation = np.where(fl_oscillation)

#                     x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
#                     grad[fl_oscillation] = grad_best[fl_oscillation].clone()

#                 k = np.maximum(k - self.size_decr, self.max_iter_min)  #

#                 # elif self.norm == "L1":
#                 #     sp_curr = L0_norm(x_best - x_nat)
#                 #     fl_redtopk = (sp_curr / sp_old) < 0.95
#                 #     topk = sp_curr / n_fts / 1.5
#                 #     step_size[fl_redtopk] = alpha * self.epsilon
#                 #     step_size[~fl_redtopk] /= adasp_redstep
#                 #     step_size.clamp_(
#                 #         alpha * self.epsilon / adasp_minstep, alpha * self.epsilon
#                 #     )
#                 #     sp_old = sp_curr.clone()

#                 #     x_adv[fl_redtopk] = x_best[fl_redtopk].clone()
#                 #     grad[fl_redtopk] = grad_best[fl_redtopk].clone()

#                 counter3 = 0
#     logger.info(f"\n final loss:{loss:.3f}")
#     self.check_feasibility(x_best.cpu())
#     return x_best.cpu(), x_best.cpu(),dict(), 0, 0, pred.cpu()

# def check_oscillation(self, x, j, k, y5, k3=0.75):
#     """
#     Parameters
#     ----------
#     x: ndarray
#         loss of criterion, which shape is (the number of iteration, the number of batch).
#     j: int
#         iteration
#     k: int
#         checkpointiteration.
#     y5: ndarray
#         the best loss of criterion, which shape is (the number of batch)
#     k3: float
#         rho

#     Returns
#     -------

#     """
#     # condition1
#     t = np.zeros(x.shape[1])
#     for counter5 in range(k):
#         # 1_{f(x^(i+1)) > f(x^(i))}
#         t += x[j - counter5] > x[j - counter5 - 1]  # .
#     return t <= k * k3 * np.ones(t.shape)

# @torch.no_grad()
# def set_bounds(self, x_nat):
#     self.upper = (x_nat + self.epsilon).clamp(0, 1).clone().to(self.device)
#     self.lower = (x_nat - self.epsilon).clamp(0, 1).clone().to(self.device)

# @torch.no_grad()
# def set_projection(self, x_nat: torch.Tensor):
#     from core.projection import ProjectionLinf
#     self.set_bounds(x_nat)
#     self.projection = ProjectionLinf(lower=self.lower, upper=self.upper)

# def check_feasibility(self, x: torch.Tensor):
#     assert (x >= self.lower.cpu()).all()
#     assert (x <= self.upper.cpu()).all()
