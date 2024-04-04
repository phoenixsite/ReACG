import os

from robustbench.model_zoo import model_dicts
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from robustbench.utils import load_model
from tqdm import tqdm

for dataset in ["cifar10", "cifar100", "imagenet"]:
    for threatmodel in ["Linf", "L2"]:
        try:
            models = model_dicts[BenchmarkDataset(dataset)][
                ThreatModel(threatmodel)
            ].keys()
        except:
            continue
        for model_name in tqdm(models):
            if os.path.isfile(f"models/{dataset}/Linf/{model_name}.pt"):
                continue
            try:
                print(model_name)
                model = load_model(
                    model_name, dataset=dataset, threat_model=threatmodel
                )
                # model.eval()
            except:
                print("failed:", model_name)
                os.remove(f"models/{dataset}/Linf/{model_name}.pt")
