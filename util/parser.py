import argparse
from os.path import join

import yaml


def parse_arguments():
    parser = argparse.ArgumentParser(description="Machine Learning Model")
    parser.add_argument("--config", type=str, default=join("configs", "default_linprobe.yaml"),
                        help="Path to the configuration file")
    args = parser.parse_args()
    return args


def read_config_file(config_file, default_file="configs/default_config.yaml"):
    # Chargement des valeurs par défaut
    with open(default_file, "r") as f:
        default_config = yaml.safe_load(f) or {}

    # Chargement des valeurs spécifiques à l'expérience
    with open(config_file, "r") as f:
        custom_config = yaml.safe_load(f) or {}

    # Fusion : les valeurs spécifiques écrasent les valeurs par défaut
    merged_config = {**default_config, **custom_config}

    return DotDict(merged_config)

class DotDict(dict):
    """Dot notation access to dictionary attributes."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self): return self.__dict__

    def __setstate__(self, d): self.__dict__.update(d)
    

def extract_mask_ratios_from_path(path):
    mask_ratio_mae = path.split("/")[-3].split("_")[-2]
    mask_ratio_jigsaw = path.split("/")[-3].split("_")[-1]
    w_jigsaw = path.split("/")[-2].split("_")[-2]
    w_siam = path.split("/")[-2].split("_")[-1]
    return mask_ratio_mae, mask_ratio_jigsaw, w_jigsaw, w_siam

