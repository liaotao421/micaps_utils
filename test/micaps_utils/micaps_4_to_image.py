# micaps4转化为图片

import cv2
from .micaps_data_io import read_micaps_4
import numpy as np
from PIL import Image


# 按论文中的方法生成图片
def generate_image(filepath, image_size):

    array, lon_range, lat_range = read_micaps_4(filepath)
    new_img = cv2.resize(array, image_size)
    new_grid = new_img / 4
    new_grid = np.uint8(new_grid)

    pixel = ((new_grid - new_grid.min()) / (new_grid.max() - new_grid.min())) * 255
    image = Image.fromarray(pixel)
    return image


# image = generate_image('./sample_data/micaps_4/19020108.000', (512, 320))
# image.show()
