import math
from typing import Tuple

import torch
import torch.nn as nn

from utils import setup_logger

logger = setup_logger(__name__)

# def updateParam(obj : Dict, new_param : Dict): # obj.update(new_param)でいいのでは？
#     for key in new_param:
#         setattr(obj, key, new_param[key])


@torch.no_grad()
def compute_accuracy(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    model: nn.Module,
    device: torch.device,
    K: int = 9,
) -> Tuple[torch.Tensor]:
    """
    Calculate the accuracy of the model according to the samples
    `x` and the labels `y`.

    :param x: Samples.
    :param y: Labels of each ``x``.
    :param batch_size: 
    :param model: Model used to classify the data in ``x``.
    :param device: Hardware where the operations are done.
    :param K: 
    """

    n_examples = len(x)
    acc = torch.ones((n_examples,), dtype=bool)
    cw_loss = torch.ones((n_examples,))
    nbatches = math.ceil(n_examples / batch_size)
    target_image_indices_all = torch.arange(n_examples, dtype=torch.long)
    y_target = torch.ones((n_examples, K), dtype=torch.long)

    for idx in range(nbatches):
        begin = idx * batch_size
        end = min((idx + 1) * batch_size, n_examples)
        target_image_indices = target_image_indices_all[begin:end]
        logit = model(x[target_image_indices].clone().to(device)).cpu()
        preds = logit.argmax(1)
        acc[target_image_indices] = preds == y[target_image_indices]
        inds = logit.argsort(1)
        cw_loss[target_image_indices] = (
            logit[torch.arange(end - begin), inds[:, -2]]
            * acc[target_image_indices].float()
            + logit[torch.arange(end - begin), inds[:, -1]]
            * (1.0 - acc[target_image_indices].float())
            - logit[torch.arange(end - begin), y[target_image_indices]]
        )
        for _k in range(2, K + 2):
            y_target[target_image_indices, _k - 2] = inds[:, -_k]
            _inds = (
                y_target[target_image_indices, _k - 2] == y[target_image_indices]
            ).to(torch.bool)
            y_target[target_image_indices, _k - 2][_inds] = inds[_inds][:, -_k]
    logger.info(f"accuracy: {acc.sum().item() / acc.shape[0] * 100:.2f}%")
    return acc, cw_loss, y_target


@torch.no_grad()
def get_beta(
    sk_1: torch.Tensor,
    gradk_1: torch.Tensor,
    gradk: torch.Tensor,
    method: str = "HS",
    use_clamp: bool = False,
) -> torch.Tensor:
    """Compute beta for CG method.

    Parameters
    ----------
    method : str, optional
        How to compute beta, by default "HS"
    use_clamp : bool, optional
        Clamp beta to [0, 1] if True, by default False

    Returns
    -------
    torch.Tensor
        beta

    Raises
    ------
    NotImplementedError
        if compute method is not implemented, raise an error.
    """
    bs = gradk.shape[0]
    _sk_1 = sk_1.reshape(bs, -1)
    _gradk = -gradk.reshape(bs, -1)
    _gradk_1 = -gradk_1.reshape(bs, -1)
    yk = _gradk - _gradk_1
    if method == "HS":
        betak = -(_gradk * yk).sum(dim=1) / (_sk_1 * yk).sum(dim=1)
    elif method == "NHS":
        t = 1.0
        denominator = (_sk_1 * yk).sum(dim=1)
        inds = denominator < t * _sk_1.norm(p=2, dim=1)
        denominator[inds] = t * _sk_1.norm(p=2, dim=1)[inds].clone()
        betak = -(_gradk * yk).sum(dim=1) / denominator
    elif method == "CD":
        betak = -_gradk.norm(p=2, dim=1) / (_sk_1 * _gradk_1).sum(dim=1)
    elif method == "FR":
        betak = gradk.norm(p=2, dim=(1, 2, 3)).pow(2) / gradk_1.norm(
            p=2, dim=(1, 2, 3)
        ).pow(2)
    elif method == "PR":
        betak = (_gradk * yk).sum(dim=1) / gradk_1.norm(p=2, dim=(1, 2, 3)).pow(2)
    elif method == "DY":
        betak = gradk.norm(p=2, dim=(1, 2, 3)).pow(2) / (_sk_1 * yk).sum(dim=1)
    elif method == "HZ":
        betak = (
            (
                yk
                - 2
                * _sk_1
                * (yk.norm(p=2).pow(2) / (_sk_1 * yk).sum(dim=1)).unsqueeze(-1)
            )
            * _gradk
        ).sum(dim=1) / (_sk_1 * yk).sum(dim=1)
    elif method == "DL":
        betak = ((yk - _sk_1) * _gradk).sum(dim=1) / (_sk_1 * yk).sum(dim=1)
    elif method == "LS":
        betak = -(_gradk * yk).sum(dim=1) / (_sk_1 * _gradk_1).sum(dim=1)
    elif method == "D": # Daniel, 1967 の式に基づく。単位行列でヘッセ行列を近似した版.。
        betak = (_gradk * _sk_1).sum(dim=1) / sk_1.norm(p=2, dim=(1, 2, 3)).pow(2)
    elif method == "RMIL":
        betak = (_gradk * yk).sum(dim=1) / sk_1.norm(p=2, dim=(1, 2, 3)).pow(2)
    elif method == "RMIL+":
        betak = (_gradk * (yk - _sk_1)).sum(dim=1) / sk_1.norm(p=2, dim=(1, 2, 3)).pow(
            2
        )
    else:
        method_tmp = method.split("-")
        if len(method_tmp) == 2:
            beta1, beta2 = method_tmp
            betak1 = get_beta(sk_1, gradk_1, gradk, beta1, use_clamp).squeeze()
            betak2 = get_beta(sk_1, gradk_1, gradk, beta2, use_clamp).squeeze()
            overwrite_inds = betak1.abs() >= betak2.abs()
            betak1[overwrite_inds] = betak2[overwrite_inds].clone()
            betak = betak1.clone()
        else:
            raise NotImplementedError(f"{method} is not implemented.")

    if use_clamp:
        betak = betak.clamp(min=0)
    infty_inds = torch.isinf(betak)
    if infty_inds.any():
        logger.warning(f"#infty occurs: {infty_inds.sum().item()} times")
        if infty_inds.sum().item() == bs:
            logger.warning("infty values are rounded.")
            betak[infty_inds] = 0.0
        else:
            betak[infty_inds] = (
                get_beta(
                    sk_1[infty_inds].double(),
                    gradk_1[infty_inds].double(),
                    gradk[infty_inds].double(),
                    method,
                    use_clamp,
                )
                .reshape(*betak[infty_inds].shape)
                .float()
            )

    logger.info(f"Number of #nan values in beta: {torch.isnan(betak).sum().item()}")
    betak[torch.isnan(betak)] = 0.0
    if len(betak.shape) == 1:
        betak = betak.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    return betak


def normalize(vec: torch.Tensor, norm: str = "sign") -> torch.Tensor:
    if norm == "sign":
        return torch.sign(vec)
    elif norm == "l2":
        return vec / (vec.norm(p=2, dim=(1, 2, 3), keepdim=True) + 1e-12)
    elif norm == "l1":
        return vec / (vec.norm(p=1, dim=(1, 2, 3), keepdim=True) + 1e-12)
    elif norm == "linf":
        return vec / (vec.norm(p=torch.inf, dim=(1, 2, 3), keepdim=True) + 1e-12)
    elif norm == "vods":
        return vec / (vec.norm(p=2, dim=(1, 2, 3), keepdim=True).pow(2) + 1e-12)
    elif norm == "softsign":
        vec = vec / vec.norm(p=torch.inf, dim=(1, 2, 3), keepdim=True)
        softsign_vec = vec / (1 + vec.abs())
        return softsign_vec
    elif norm == "tmp":
        # softsign_vec = vec / (1 + vec.abs())
        # return (1 - softsign_vec.abs()) * vec.sign()
        # print(vec.mean(dim=(1, 2, 3)), vec.std(dim=(1, 2, 3)))
        return vec.sign() / (1 + vec.abs())
    elif norm == "tmp2":  # Good for ACG, but the reason is not sure
        # import matplotlib.pyplot as plt
        # ave = vec.mean(dim=(1, 2, 3)).cpu().detach().numpy()
        # std = vec.std(dim=(1, 2, 3)).cpu().detach().numpy()
        # plt.plot(ave)
        # ax = plt.twinx()
        # ax.fill_between(range(vec.shape[0]), ave-std, ave+std, color="C1")
        # plt.savefig("../tmp.png")
        # plt.clf()
        # plt.close()
        # breakpoint()
        normalized_vec = vec.sign() + vec
        normalized_vec /= normalized_vec.norm(p=torch.inf, dim=(1, 2, 3), keepdim=True)
        return normalized_vec
    elif norm == "tmp3":
        normalized_vec = vec / (
            vec.norm(p=2, dim=(1, 2, 3), keepdim=True).pow(2) + 1e-12
        )
        return normalized_vec.sign()
    else:
        raise NotImplementedError(f"{norm} is not implemented.")
