import argparse
import os
import warnings

import numpy as np
import yaml
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure
from powdiffrac.simulation import Powder
from tqdm import tqdm

from data_transformation import generate_variations


def main(
    system_name: str,
    file_path: str,
    config_path: str,
    # num variations
    n_train: int,
    n_val: int,
):
    if not os.path.exists(system_name):
        os.mkdir(system_name)
    # else:
    #     raise ValueError("Directory already exists, please choose a different name")

    # parse config
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    two_theta = config["two_theta"]
    step_size = config["step_size"]
    seed = config["seed"]
    steps = np.arange(two_theta[0], two_theta[1], step_size)

    struct = Structure.from_file(file_path)
    powder = Powder(
        struct,
        two_theta=two_theta,
        step_size=step_size,
        vary_strain=True,
        vary_texture=True,
        max_texture=config["texture"],
        max_strain=config["strain"],
        seed=seed
    )
    powder.calculator = XRDCalculator(config["wavelength"])
    # ignore warnings from pymatgen and py_powder_diffraction
    warnings.filterwarnings("ignore", category=UserWarning)
    powder.domain_size = 0

    print("Generating training data")
    x_train = np.zeros([n_train, steps.size])
    for n in tqdm(range(n_train)):
        x_train[n] = powder.get_signal(vary=True)[: steps.size]

    if config["transmission"]:
        # correct transmission mode
        x_train *= np.linspace(1, 0.5, steps.size)[None, :]

    peak_range = config["multiphase_peak_range"]
    if peak_range:
        # eval indices for min,max of multi-phase peak range
        peak_range = tuple([np.argmin(np.abs(t - steps)) for t in peak_range])
    holder = config["sample_holder"]
    holder = np.argmin(np.abs(holder - steps)) if holder else None

    xt, yt = generate_variations(
        x_train,
        step_size=step_size,
        background_type=config["background_type"],
        multi_peak_range=peak_range,
        max_multi_peak=config["multiphase_peak_height"][1],
        detection_threshold=config["multiphase_peak_height"][0],
        num_multi_peaks=config["multiphase_peak_num"],
        restricted_area=config["multiphase_restricted"],
        fwhm_range=config["fwhm_range"],
        noise_lvl=config["noise_level"],
        sample_holder=holder,
        random_shift=config["random_shift"],
        seed=seed,
    )

    np.save(os.path.join(system_name, "x_train.npy"), xt)
    np.save(os.path.join(system_name, "y_train.npy"), yt)

    print("Generating validation data")
    x_val = np.zeros([n_val, steps.size])
    for n in tqdm(range(n_val)):
        x_val[n] = powder.get_signal(vary=True)[: steps.size]

    if config["transmission"]:
        # correct transmission mode
        x_val *= np.linspace(1, 0.5, steps.size)[None, :]

    xv, yv = generate_variations(
        x_val,
        step_size=step_size,
        background_type=config["background_type"],
        multi_peak_range=peak_range,
        max_multi_peak=config["multiphase_peak_height"][1],
        detection_threshold=config["multiphase_peak_height"][0],
        num_multi_peaks=config["multiphase_peak_num"],
        restricted_area=config["multiphase_restricted"],
        fwhm_range=config["fwhm_range"],
        noise_lvl=config["noise_level"],
        sample_holder=holder,
        random_shift=config["random_shift"],
        seed=seed,
    )

    np.save(os.path.join(system_name, "x_val.npy"), xv)
    np.save(os.path.join(system_name, "y_val.npy"), yv)

    np.save(os.path.join(system_name, "steps.npy"), steps)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate xrd signals")
    parser.add_argument(
        "system_name", type=str, nargs="?", help="name of the system to simulate"
    )
    parser.add_argument(
        "--file_path",
        type=str,
        help="path to cif file containing structure",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./config.yml",
        help="path to config yaml file",
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=1000,
        help="number of variations per phase for training data",
    )
    parser.add_argument(
        "--n_val",
        type=int,
        default=300,
        help="number of variations per phase for training data",
    )

    args = parser.parse_args()
    main(**vars(args))
