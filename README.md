# Rescaling ACG

## Requirements

- Python >= 3.11, <3.12
- CUDA >= 11.8
- GCC >= 4.5
- poetry >= 1.4.2

## Installation

### Poetry

```
curl -sSL https://install.python-poetry.org | python3 -
```

#### Specify the python-version

```
poetry env use <python-version>
```

If you use `pyenv`, you have to run `pyenv local <python-version>` in advance.

### Dependencies

```
poetry install
```

## Dataset

* imagenet
  1. `cd data/imagenet`
  2. Download `ILSVRC2012_img_val.tar` and `ILSVRC2012_devkit_t12.tar.gz` from [ImageNet official site](https://image-net.org/index.php)
  3. `mkdir val && tar -xf ILSVRC2012_img_val.tar -C ./val`
  4. `tar -xzf ILSVRC2012_devkit_t12.tar.gz`
  5. `python build_dataset.py`
  6. `mv val val_original && mv ILSVRC2012_img_val_for_ImageFolder val`

## Usage

### Run attacks

#### APGD

```bash
poetry run python -B run_evaluation.py -p ../configs/config_apgd.yaml -g 0 -o ../result --log_level 20 --cmd_param attacker_name:str:APGD 
```

#### ACG

```bash
poetry run python -B run_evaluation.py -p ../configs/config_acg.yaml -g 0 -o ../result --log_level 20 --cmd_param attacker_name:str:ACG 
```

#### ACG+T

```bash
poetry run python -B run_evaluation.py -p ../configs/config_acg_t.yaml -g 0 -o ../result --log_level 20 --cmd_param attacker_name:str:ACG 
```

#### ACG+R

```bash
poetry run python -B run_evaluation.py -p ../configs/config_acg_r.yaml -g 0 -o ../result --log_level 20 --cmd_param attacker_name:str:ReACG 
```

#### ReACG

```bash
poetry run python -B run_evaluation.py -p ../configs/config_reacg.yaml -g 0 -o ../result --log_level 20 --cmd_param attacker_name:str:ReACG 
```

### Models used in the experiments

`selected_models.yaml` shows the model names used in the experiments.
You can generate adversarial examples for `<model_name>` by specifying `model_name:str:<model_name>` after `--cmd_param`. Similarly, you can specify the dataset and batch size as `dataset:str:<dataset>`, `batch_size:int:<batch_size>`.

`config/config_<attacker_name>.yaml` specifies the hyperparameters of the attack `<attacker_name>`. You can set different parameters by editing this file.
