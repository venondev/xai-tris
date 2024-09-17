from typing import List, Dict

import numpy as np
import torch
import random
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import StratifiedShuffleSplit

from xai_master.scenarios.xai_tris.xai_tris_repo.common import (
    DataRecord,
    DataScenarios,
    SEED,
)
from xai_master.scenarios.xai_tris.xai_tris_repo.utils import (
    load_json_file,
    dump_as_pickle,
)
from xai_master.scenarios.xai_tris.xai_tris_repo.data.data_utils import (
    generate_backgrounds,
    generate_imagenet,
    generate_fixed,
    generate_translations_rotations,
    generate_xor,
    normalise_data,
    scale_to_bound,
)

from datetime import datetime
from pathlib import Path

import sys


scenario_dict = {
    "linear": generate_fixed,
    "multiplicative": generate_fixed,
    "translations_rotations": generate_translations_rotations,
    "xor": generate_xor,
}


def data_generation_process(config: Dict, output_dir: str):
    # experiment_list = []
    image_shape = np.array(config["image_shape"]) * config["image_scale"]
    for i in range(config["num_experiments"]):  # generate multiple datasets if desired
        backgrounds = generate_backgrounds(
            config["sample_size"], config["mean_data"], config["var_data"], image_shape
        )
        if config["use_imagenet"]:
            imagenet_backgrounds = generate_imagenet(config["sample_size"])

        # Iterate over config
        k = 0
        for scenario, params in config["parameterizations"].items():
            for pattern_scale in params["pattern_scales"]:
                config["pattern_scale"] = pattern_scale
                patterns = scenario_dict.get(scenario)(
                    params=config, image_shape=list(image_shape)
                )
                ground_truths = patterns.copy()

                for correlated in ["white", "correlated", "imagenet"]:
                    if correlated == "imagenet" and not config["use_imagenet"]:
                        continue
                    copy_backgrounds = np.zeros(
                        (config["sample_size"], image_shape[0] * image_shape[1])
                    )
                    params["correlated_background"] = correlated
                    if correlated == "correlated":
                        for j, background in enumerate(backgrounds.copy()):
                            copy_backgrounds[j] = gaussian_filter(
                                np.reshape(
                                    background, (image_shape[0], image_shape[1])
                                ),
                                config["smoothing_sigma"],
                            ).reshape((image_shape[0] * image_shape[1]))

                            if "correlated_additive_noise" in params:
                                noise = np.random.normal(
                                    config["mean_data"],
                                    config["var_data"],
                                    (image_shape[0], image_shape[1]),
                                )
                                copy_backgrounds[j] += (
                                    noise.reshape((image_shape[0] * image_shape[1]))
                                    * params["correlated_additive_noise"]
                                )

                        alpha_ind = 1
                    elif correlated == "imagenet" and config["use_imagenet"]:
                        copy_backgrounds = imagenet_backgrounds.copy()
                        alpha_ind = 2
                    else:
                        copy_backgrounds = backgrounds.copy()
                        alpha_ind = 0

                    for alpha in params["snrs"][alpha_ind]:
                        print(
                            f"Generating scenario {scenario} with {correlated} background and alpha={alpha}"
                        )
                        copy_patterns = patterns.copy()
                        if params["manipulation_type"] == "multiplicative":
                            copy_patterns = 1 - alpha * copy_patterns

                        normalised_patterns, normalised_backgrounds = normalise_data(
                            copy_patterns, copy_backgrounds
                        )

                        if params["manipulation_type"] == "multiplicative":
                            x = normalised_patterns * normalised_backgrounds
                        else:
                            x = (
                                alpha * normalised_patterns.copy()
                                + (1 - alpha) * normalised_backgrounds.copy()
                            )

                        scale = 1 / np.max(np.abs(x))
                        x = np.apply_along_axis(scale_to_bound, 1, x, scale)

                        class_0_labels = torch.as_tensor(
                            [[0]] * int(config["sample_size"] / len(config["patterns"]))
                        )
                        class_1_labels = torch.as_tensor(
                            [[1]] * int(config["sample_size"] / len(config["patterns"]))
                        )

                        x = torch.as_tensor(x, dtype=torch.float16)
                        y = torch.ravel(torch.cat((class_0_labels, class_1_labels)))

                        # 90/5/5 split for 64x64 data, 80/10/10 split for 8x8
                        data_splitter = StratifiedShuffleSplit(
                            n_splits=1,
                            test_size=config["test_split"],
                            random_state=SEED,
                        )
                        train_indices, val_indices = list(
                            data_splitter.split(X=x, y=y)
                        )[0]

                        x_train = x[train_indices]
                        y_train = y[train_indices]
                        x_val_test = x[val_indices]
                        y_val_test = y[val_indices]

                        masks_train = ground_truths[train_indices]
                        masks_val_test = ground_truths[val_indices]

                        data_splitter = StratifiedShuffleSplit(
                            n_splits=1, test_size=0.5, random_state=SEED
                        )
                        val_indices, test_indices = list(
                            data_splitter.split(X=x_val_test, y=y_val_test)
                        )[0]

                        x_val = x_val_test[val_indices]
                        y_val = y_val_test[val_indices]
                        masks_val = masks_val_test[val_indices]

                        x_test = x_val_test[test_indices]
                        y_test = y_val_test[test_indices]
                        masks_test = masks_val_test[test_indices]

                        correlated_string = "uncorrelated"
                        if correlated == "imagenet":
                            correlated_string = "imagenet"
                        elif correlated == "correlated":
                            correlated_string = "correlated"

                        scenario_key = f'{params["scenario"]}_{config["image_scale"]}d{config["pattern_scale"]}p_{alpha}_{correlated_string}'

                        scenario = {
                            "key": scenario_key,
                            "x_train": x_train,
                            "y_train": y_train,
                            "x_val": x_val,
                            "y_val": y_val,
                            "x_test": x_test,
                            "y_test": y_test,
                            "masks_train": masks_train,
                            "masks_val": masks_val,
                            "masks_test": masks_test,
                        }

                        # Save as torch dict
                        torch.save(scenario, f"{output_dir}/{scenario_key}.pt")


def main():
    config_file = "data_config"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    config = load_json_file(file_path=f"data/{config_file}.json")

    date = datetime.now().strftime("%Y-%m-%d-%H-%M")
    folder_path = f'{config["output_dir"]}/{date}'
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    data_generation_process(config=config, output_dir=folder_path)


if __name__ == "__main__":
    main()
