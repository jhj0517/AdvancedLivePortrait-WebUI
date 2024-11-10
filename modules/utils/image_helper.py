from PIL import Image
import torch
import numpy as np


class PreparedSrcImg:
    def __init__(self, src_rgb, crop_trans_m, x_s_info, f_s_user, x_s_user, mask_ori):
        self.src_rgb = src_rgb
        self.crop_trans_m = crop_trans_m
        self.x_s_info = x_s_info
        self.f_s_user = f_s_user
        self.x_s_user = x_s_user
        self.mask_ori = mask_ori


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def rgb_crop(rgb, region):
    return rgb[region[1]:region[3], region[0]:region[2]]


def rgb_crop_batch(rgbs, region):
    return rgbs[:, region[1]:region[3], region[0]:region[2]]


def get_rgb_size(rgb):
    return rgb.shape[1], rgb.shape[0]


def create_transform_matrix(x, y, s_x, s_y):
    return np.float32([[s_x, 0, x], [0, s_y, y]])


def calc_crop_limit(center, img_size, crop_size):
    pos = center - crop_size / 2
    if pos < 0:
        crop_size += pos * 2
        pos = 0

    pos2 = pos + crop_size

    if img_size < pos2:
        crop_size -= (pos2 - img_size) * 2
        pos2 = img_size
        pos = pos2 - crop_size

    return pos, pos2, crop_size


def save_image(numpy_array: np.ndarray, output_path: str):
    out = Image.fromarray(numpy_array)
    out.save(output_path, compress_level=1, format="png")
    return output_path


def image_path_to_array(image_path: str) -> np.ndarray:
    image = Image.open(image_path)
    image_array = np.array(image)
    if len(image_array.shape) <= 3:
        image_array = image_array[np.newaxis, ...]

    return image_array
