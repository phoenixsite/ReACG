import gc
import sys

# import time
import torch
import yaml

sys.path.append("src")
from robustbench.model_zoo import model_dicts
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel

from utils.loader import load_model_and_dataset

# from torchsummary import summary

device = torch.device("cuda:0")
try:
    with open("batchsize.yaml", "r") as f:
        data = yaml.safe_load(f)
except FileNotFoundError:
    data = dict()
for dataset in ["cifar10", "cifar100", "imagenet"]:
    if dataset not in data:
        data[dataset] = dict()
    threat_models = ["Linf", "L2"] if dataset == "cifar10" else ["Linf"]
    n_examples = 5000 if dataset == "imagenet" else 10000
    for threat_model in threat_models:
        models = model_dicts[BenchmarkDataset(dataset)][
            ThreatModel(threat_model)
        ].keys()
        if threat_model not in data[dataset]:
            data[dataset][threat_model] = dict()
        for model_name in models:
            if model_name in data[dataset][threat_model]:
                continue
            torch.cuda.empty_cache()
            gc.collect()
            dic = dict()
            model, x, y = load_model_and_dataset(
                model_name,
                dataset=dataset,
                n_examples=n_examples,
                threat_model=threat_model,
                relative_root="./",
            )
            model = model.to(device)
            model.eval()
            left = 0
            right = n_examples + 5
            while left + 1 < right:
                torch.cuda.empty_cache()
                gc.collect()
                logit = None
                loss = None
                grad = None
                try:
                    bs = (left + right) // 2
                    _x = x[:bs].clone().to(device)
                    _x.requires_grad_()
                    logit = model(_x)
                    loss = logit[:, -1] - logit[:, -2]
                    grad = torch.autograd.grad(loss.sum(), [_x])[0].detach().clone()
                    left = bs
                except Exception as err:
                    right = bs
                if logit is not None:
                    del logit
                if loss is not None:
                    del loss
                if grad is not None:
                    del grad
                torch.cuda.empty_cache()
                gc.collect()
            _bs = int(0.7 * left)
            if _bs > 10:
                _bs = int(_bs / 10) * 10
            data[dataset][threat_model][model_name] = _bs
            del model
            del x
            del y
            torch.cuda.empty_cache()
            gc.collect()
            # summary(model, x.squeeze().shape)
            # time.sleep(5)

with open("batchsize.yaml", "w") as f:
    yaml.dump(data, f)
