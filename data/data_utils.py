import numpy as np
from typing import List, Dict, Tuple
from glob import glob
import random
from PIL import Image
from scipy.ndimage import gaussian_filter
from xai_master.scenarios.xai_tris.xai_tris_repo.common import SEED
from matplotlib import pyplot as plt

np.random.seed(SEED)

import os

os.environ["PYTHONHASHSEED"] = str(SEED)

import random

random.seed(SEED)


def get_patterns(params: Dict) -> List:
    manip = params["manipulation"]
    scale = params["pattern_scale"]
    t = np.array([[manip, 0], [manip, manip], [manip, 0]])

    l = np.array([[manip, 0], [manip, 0], [manip, manip]])

    pattern_dict = {
        "t": np.kron(t, np.ones((scale, scale))),
        "l": np.kron(l, np.ones((scale, scale))),
    }

    chosen_patterns = []
    for pattern_name in params["patterns"]:
        chosen_patterns.append(pattern_dict[pattern_name])
    return chosen_patterns


def normalise_data(
    patterns: np.array, backgrounds: np.array
) -> Tuple[np.array, np.array]:
    patterns /= np.linalg.norm(patterns, ord="fro")
    d_norm = np.linalg.norm(backgrounds, ord="fro")
    distractor_term = backgrounds if 0 == d_norm else backgrounds / d_norm
    return patterns, distractor_term


def scale_to_bound(row, scale):
    # scale = bound / np.max(np.abs(row))
    return row * scale


from scipy.ndimage import gaussian_filter1d


def apply_gaussian_filter_no_padding(image, sigma):
    """
    Apply a Gaussian filter to an image without padding.
    Out-of-bound pixels are excluded, and the filter is normalized at edges.
    """
    # Create the 2D Gaussian kernel
    size = int(2 * np.ceil(3 * sigma) + 1)  # Kernel size based on sigma
    x = np.arange(-size // 2 + 1, size // 2 + 1)
    gaussian_kernel = np.exp(-0.5 * (x / sigma) ** 2)
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

    # Prepare the output array
    filtered_image = np.zeros_like(image)

    # Iterate over each pixel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Determine the bounds of the region for the kernel
            x_min = max(0, i - size // 2)
            x_max = min(image.shape[0], i + size // 2 + 1)
            y_min = max(0, j - size // 2)
            y_max = min(image.shape[1], j + size // 2 + 1)

            # Extract the region of interest
            region = image[x_min:x_max, y_min:y_max]

            # Adjust the kernel to match the region size
            x_kernel_start = max(0, size // 2 - i)
            x_kernel_end = x_kernel_start + (x_max - x_min)
            y_kernel_start = max(0, size // 2 - j)
            y_kernel_end = y_kernel_start + (y_max - y_min)

            adjusted_kernel = gaussian_kernel[x_kernel_start:x_kernel_end].reshape(
                -1, 1
            ) * gaussian_kernel[y_kernel_start:y_kernel_end].reshape(1, -1)

            # Normalize the kernel for the current region
            adjusted_kernel /= adjusted_kernel.sum()

            # Apply the filter
            filtered_image[i, j] = np.sum(region * adjusted_kernel)

    return filtered_image


def generate_backgrounds(
    sample_size: int,
    mean_data: int,
    var_data: float,
    image_shape: list = [8, 8],
    smoothing_sigma: float = 1,
) -> np.array:
    backgrounds_raw = np.random.normal(
        mean_data, var_data, size=(sample_size, image_shape[0], image_shape[1])
    )
    backgrounds_corr_old = np.zeros_like(backgrounds_raw)
    for i in range(sample_size):
        backgrounds_corr_old[i] = gaussian_filter(
            backgrounds_raw[i].copy(), smoothing_sigma
        )

    backgrounds_corr_new = np.zeros_like(backgrounds_raw)
    for i in range(sample_size):
        backgrounds_corr_new[i] = apply_gaussian_filter_no_padding(
            backgrounds_raw[i].copy(), smoothing_sigma
        )

    return (
        backgrounds_raw.reshape((sample_size, image_shape[0] * image_shape[1])),
        backgrounds_corr_old.reshape((sample_size, image_shape[0] * image_shape[1])),
        # backgrounds_corr_new.reshape((sample_size, image_shape[0] * image_shape[1])),
    )

    # cut_out_x, cut_out_y = image_shape[0] // 2, image_shape[1] // 2
    # backgrounds = backgrounds_raw[
    #     :,
    #     cut_out_x : cut_out_x + image_shape[0],
    #     cut_out_y : cut_out_y + image_shape[1],
    # ].reshape((sample_size, image_shape[0] * image_shape[1]))

    # backgrounds_corr_raw = gaussian_filter(backgrounds_raw, smoothing_sigma)
    # backgrounds_corr = backgrounds_corr_raw[
    #     :,
    #     cut_out_x : cut_out_x + image_shape[0],
    #     cut_out_y : cut_out_y + image_shape[1],
    # ].reshape((sample_size, image_shape[0] * image_shape[1]))

    return backgrounds, backgrounds_corr


def generate_imagenet(sample_size: int) -> np.array:
    image_paths = glob(f"./imagenet_images/*")
    backgrounds = np.zeros((sample_size, 64 * 64))
    new_width, new_height = 64, 64
    i = 0

    # sample from sample_size + extra so that we can skip images smaller than the resize size
    for image_path in random.sample(
        image_paths, sample_size + int(len(image_paths) / 10)
    ):
        if i == sample_size:
            return backgrounds
        img = Image.open(image_path)

        width, height = img.size  # Get dimensions
        if width < 64 or height < 64:
            continue

        if height <= width:
            scale_factor = 64 / height
        else:
            scale_factor = 64 / width

        resize = (int(scale_factor * width), int(scale_factor * height))

        img = img.resize(resize)
        width, height = img.size

        left = round((width - new_width) / 2)
        top = round((height - new_height) / 2)
        x_right = round(width - new_width) - left
        x_bottom = round(height - new_height) - top
        right = width - x_right
        bottom = height - x_bottom

        # Crop the center of the image
        img = img.crop((left, top, right, bottom))

        if img.mode != "RGB":
            img = img.convert("RGB")
        grey = np.dot(np.array(img)[..., :3], [0.299, 0.587, 0.114]).reshape((64 * 64))
        # print(grey.shape)
        # print(backgrounds.shape)
        # print(backgrounds[i].shape)
        backgrounds[i] = grey - grey.mean()
        i += 1
    return backgrounds


# def generate_imagenet(sample_size: int) -> np.array:
#     # cat_paths = glob(f'./cats_dogs/cat.*')
#     # dog_paths = glob(f'./cats_dogs/dog.*')
#     image_paths = glob(f'./cats_dogs/*')
#     backgrounds = np.zeros((sample_size, 128*128))
#     for i, image_path in enumerate(random.sample(image_paths, sample_size)):
#         img = Image.open(image_path)
#         grey = np.dot(np.array(img)[...,:3], [0.299, 0.587, 0.114]).reshape((128*128))
#         backgrounds[i] = grey - grey.mean()
#     return backgrounds


def generate_fixed(params: Dict, image_shape: list):
    patterns = np.zeros((params["sample_size"], image_shape[0], image_shape[1]))
    chosen_patterns = get_patterns(params)
    j = 0
    for k, pattern in enumerate(chosen_patterns):
        pos = params["positions"][k]
        for i in range(int(params["sample_size"] / len(params["patterns"]))):
            patterns[j][
                pos[0] : pos[0] + pattern.shape[0], pos[1] : pos[1] + pattern.shape[1]
            ] = pattern
            if params["pattern_scale"] > 3:
                patterns[j] = gaussian_filter(patterns[j], 1.5)
            j += 1

    return np.reshape(
        patterns, (params["sample_size"], image_shape[0] * image_shape[1])
    )


def generate_translations_rotations(params: Dict, image_shape: list) -> np.array:
    patterns = np.zeros((params["sample_size"], image_shape[0], image_shape[1]))
    chosen_patterns = get_patterns(params)

    j = 0
    for pattern in chosen_patterns:
        for i in range(int(params["sample_size"] / len(params["patterns"]))):
            pattern_adj = pattern

            rand = np.random.randint(0, high=4)
            if rand > 0:
                pattern_adj = np.rot90(pattern, k=rand)

            rand_y = np.random.randint(
                0, high=(image_shape[0]) - pattern_adj.shape[0] + 1
            )
            rand_x = np.random.randint(
                0, high=(image_shape[0]) - pattern_adj.shape[1] + 1
            )
            pos = (rand_y, rand_x)

            patterns[j][
                pos[0] : pos[0] + pattern_adj.shape[0],
                pos[1] : pos[1] + pattern_adj.shape[1],
            ] = pattern_adj
            if params["pattern_scale"] > 3:
                patterns[j] = gaussian_filter(patterns[j], 1.5)
            j += 1

    return np.reshape(
        patterns, (params["sample_size"], image_shape[0] * image_shape[1])
    )


def generate_xor(params: Dict, image_shape: list) -> np.array:
    patterns = np.zeros((params["sample_size"], image_shape[0], image_shape[1]))
    chosen_patterns = get_patterns(params)
    poses = params["positions"]

    manips = [
        [1, 1],
        [0, 0],
        [1, 0],
        [0, 1],
    ]

    k = 0
    for ind in range(0, params["sample_size"], int(params["sample_size"] / 4)):
        pat = np.zeros((image_shape[0], image_shape[1]))
        pat[
            poses[0][0] : poses[0][0] + chosen_patterns[0].shape[0],
            poses[0][1] : poses[0][1] + chosen_patterns[0].shape[1],
        ] = chosen_patterns[0] * manips[k][0]
        pat[
            poses[1][0] : poses[1][0] + chosen_patterns[1].shape[0],
            poses[1][1] : poses[1][1] + chosen_patterns[1].shape[1],
        ] = chosen_patterns[1] * manips[k][1]
        patterns[ind : ind + int(params["sample_size"] / 4)] = pat
        k += 1

    for j, pattern in enumerate(patterns):
        if params["pattern_scale"] > 3:
            patterns[j] = gaussian_filter(pattern, 1.5)

    return np.reshape(
        patterns, (params["sample_size"], image_shape[0] * image_shape[1])
    )
