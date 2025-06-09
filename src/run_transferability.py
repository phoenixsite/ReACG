"""
Test a sample of adversarial examples generated with an attack over a model.
"""

import os
import sys
import time
import math
import yaml

import torch
from torchvision import transforms
import timm
from timm.data.config import resolve_model_data_config
from timm.data.transforms_factory import create_transform
from argparse import ArgumentParser

from robustbench.loaders import default_loader
from robustbench.data import CustomImageFolder, get_preprocessing
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel

from utils import reproducibility, setup_logger, read_yaml

TOP_BANNER = "Transferability"

def argparser():
    
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--data-dir",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-tm",
        "--target-model",
        choices=[
            "vgg16.tv_in1k",
            "inception_v3.tv_in1k",
            "resnet50.tv2_in1k",
            "inception_v3.tf_adv_in1k",
            "inception_resnet_v2.tf_ens_adv_in1k",
            "random-padding",
            "jpeg",
            "bit-reduction",
            "feature-distillation",
            "randomized-smoothing",
            "neural-representation-purifier",
            "vgg19.tv_in1k",
            "resnet152.tv2_in1k",
            "mobilenetv2_140.ra_in1k",
        ],
        required=True,
        type=str,
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=30,
        help="10:DEBUG,20:INFO,30:WARNING,40:ERROR,50:CRITICAL",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory name (not path)",
        required=True,
        type=str,
    )
    parser.add_argument("--nthreads", type=int, default=4)
    parser.add_argument("-g", "--gpu-id", type=int, default=0)
    parser.add_argument("-bs", "--batch-size", type=int, required=True)
    parser.add_argument("--test-samples", action="store_true")
    parser.add_argument("--indices-samples", default=None, type=str)
    parser.add_argument("--test-transferability", action="store_true")
    return parser


def load_adversarial_dataset(data_dir: str):
    """
    Load the images and its classes/label from a directory structure like the
    following:
        root/211/1.png
        root/211/2872.png
        root/211/4561.png
        root/567/1275.png
        root/567/21341.png
        root/567/4513.png
    so, the following directory name to the root folder is the label
    of the images it contains, and the images file name corresponds to 
    its index in the ImageNet dataset.
    """
    
    transformations = transforms.Compose([
        transforms.ToTensor()
    ])

    classes_unique = [d.name for d in os.scandir(data_dir) if d.is_dir()]
    logger.info(f"Read {len(classes_unique)} classes")
    samples, classes, names = [], [], []

    for class_unique in classes_unique:
        class_dir = os.path.join(data_dir, class_unique)

        logger.debug(f"Reading directory {class_dir}")

        for image_path in os.scandir(class_dir):
            sample = default_loader(image_path.path)
            sample = transformations(sample)
            samples.append(sample.unsqueeze(0))
            classes.append(int(class_unique))
            names.append(int(image_path.name.split(".")[0]))

    samples = torch.vstack(samples)
    classes = torch.tensor(classes)
    logger.info(f"{len(sample)} samples have been read")
    return samples, classes, names


def load_imagenet(
        model_name: str,
        names,
        threat_model: str = "Linf",
):
    """
    Load the images from the ImageNet dataset whose index is in ``names``.
    """
    
    prepr = get_preprocessing(
        BenchmarkDataset("imagenet"), ThreatModel(threat_model), model_name, None
    )
    
    dataset = CustomImageFolder(
        "data/imagenet/val",
        transform=prepr,
    )

    x_test, y_test = [], []

    for index in names:
        x, y, _ = dataset[index]
        x_test.append(x.unsqueeze(0))
        y_test.append(y)

    x_test = torch.vstack(x_test)
    y_test = torch.tensor(y_test)

    return x_test, y_test


def load_model(
        model_name: str,
        model_dir: str=os.path.join("../models"),
):
    #transformations = None
    if timm.is_model(model_name):
        model = timm.create_model(model_name, pretrained=True)
        #data_config = resolve_model_data_config(model, use_test_size=True)
        #transformations = create_transform(**data_config, is_training=False)
    # elif model_name == "random-padding":
    
    # elif model_name == "jpeg":

    # elif model_name == "bit-reduction":

    # elif model_name == "feature-distillation":

    # elif model_name == "randomized-smoothing":
            
    # elif model_name == "neural-representation-purifier":
        
    else:
        raise ValueError(f"The value {model_name} is not a valid model name.")
    
    return model

def main(args):
    
    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if torch.cuda.is_available() and args.gpu_id is not None
        else torch.device("cpu")
    )
    batch_size = args.batch_size
    target_model = load_model(args.target_model)
    target_model = target_model.to(device)
    target_model.eval()
    adv_sample, classes, names = load_adversarial_dataset(args.data_dir)

    stime = time.time()

    output_dir = os.path.join(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    good_indices_path = os.path.join(
        output_dir,
        "good_indices.yaml",
    )

    if args.test_samples:
        # Get those real images from ImageNet that have an adversary example
        real_sample, real_classes = load_imagenet(args.target_model, names)
        nexamples = len(adv_sample)
        acc = torch.ones((nexamples,), dtype=bool)
        target_sample_indices_all = torch.arange(nexamples, dtype=torch.long)
        nbatches = math.ceil(nexamples / batch_size)

        for idx in range(nbatches):
            logger.info(msg=f"idx = {idx}")
            begin = idx * batch_size
            end = min((idx + 1) * batch_size, nexamples)
            target_sample_indices = target_sample_indices_all[begin:end]
            logit = target_model(real_sample[target_sample_indices].clone().to(device)).cpu()
            preds = logit.argmax(1)
            acc[target_sample_indices] = preds == real_classes[target_sample_indices]

        clean_accuracy = acc.sum().item() / acc.shape[0] * 100
        logger.info(f"clean accuracy on real images from the adversarial dataset: {clean_accuracy:.2f}%")

        good_indices = [index for index in names if acc[index]]
        with open(good_indices_path, "w") as file:
            yaml.dump({"indices": good_indices}, file)

    else:
        good_indices = torch.tensor(read_yaml(good_indices_path).indices)
    
    if args.test_transferability:
        
        # Get those adversarial images whose original image is correcly
        # classified by the target model
        indices = (names == good_indices).nonzero()
        adv_sample = adv_sample[indices]
        classes = classes[indices]
        
        nexamples = len(adv_sample)
        acc = torch.ones((nexamples,), dtype=bool)
        target_sample_indices_all = torch.arange(nexamples, dtype=torch.long)
        nbatches = math.ceil(nexamples / batch_size)

        for idx in range(nbatches):
            logger.info(msg=f"idx = {idx}")
            begin = idx * batch_size
            end = min((idx + 1) * batch_size, nexamples)
            target_sample_indices = target_sample_indices_all[begin:end]
            logit = target_model(adv_sample[target_sample_indices].clone().to(device)).cpu()
            preds = logit.argmax(1)
            acc[target_sample_indices] = preds == classes[target_sample_indices]
            #inds = logit.argsort(1)

        logger.info(f"accuracy: {acc.sum().item() / acc.shape[0] * 100:.2f}%")

        accuracy = 100 * acc.sum().item() / acc.shape[0]
        attack_success_rate = 100 - accuracy
        msg = (
            f"adversarial images:{args.data_dir}\n"
            f"target model = {args.target_model}\n"
            f"total time (sec) = {time.time() - stime:.3f}\n"
            f"transferability ASR(%) = {attack_success_rate:.2f}\n"
        )

        
        short_summary_path = os.path.join(output_dir, f"{args.target_model}.txt")
        with open(short_summary_path, "a") as f:
            f.write(msg)
            f.write("\n")

        logger.info(msg)


if __name__ == '__main__':

    sys.path.append("../src")
    reproducibility()
    parser = argparser()
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("The batch size must be greater than 1.")
    
    if not args.test_samples and not args.indices_samples:
        raise ValueError(
            "If we are not going to obtain the images from data_dir" \
            "that are correctly classified by the the target model, then you need to provide a file with" \
            "the indices (name of the images in the data_dir) with the --indices-samples option."
        )

    logger = setup_logger.setLevel(args.log_level)
    torch.set_num_threads(args.nthreads)
    os.environ["OMP_NUM_THREADS"] = str(args.nthreads)

    main(args)