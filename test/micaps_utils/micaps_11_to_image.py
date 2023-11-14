# 读取micaps_11类数据变化为训练数据
import cv2
import numpy as np
from PIL import Image
import micaps_data_io as micaps_data_io


# 计算风速speed和风向Direction
def compute_speed_and_direction(array):
    """

    :param array: (320, 512, 2) 数组
    :return: speed 速度
            direction 方向
    """
    row, col, _ = array.shape

    u_array = array[:, :, 0]
    v_array = array[:, :, 1]

    speed = np.sqrt(u_array ** 2 + v_array ** 2)

    speed = np.where(speed >= 40, 40, speed)

    direction = np.zeros((row, col))

    for i in range(row):
        for j in range(col):
            u_value, v_value = u_array[i, j], v_array[i, j]

            if u_value == 0 and v_value >= 0:
                direction[i, j] = 0
            elif u_value > 0 and v_value == 0:
                direction[i, j] = 90
            elif u_value < 0 and v_value == 0:
                direction[i, j] = 270
            elif u_value == 0 and v_value < 0:
                direction[i, j] = 180
            elif u_value > 0 and v_value > 0:
                direction[i, j] = np.arctan(u_value / v_value) * 180 / np.pi
            elif u_value < 0 and v_value > 0:
                direction[i, j] = np.arctan(-v_value / u_value) * 180 / np.pi + 270
            elif u_value < 0 and v_value < 0:
                direction[i, j] = np.arctan(u_value / v_value) * 180 / np.pi + 180
            elif u_value > 0 and v_value < 0:
                direction[i, j] = np.arctan(-v_value / u_value) * 180 / np.pi + 90

    return speed, direction


# 计算rgb三个通道的值
def compute_rbg(speed, direction, only_r=False):
    '''

    :param only_r: 仅需要R通道（表示分速大小）
    :param speed: 速度
    :param direction: 方向
    :return: image：rgb三个通道合并后的图片
    '''
    row, col = speed.shape

    result_image = np.zeros((row, col, 3), dtype=np.uint8)

    # 计算R通道
    result_image[..., 0] = np.where(speed < 14, 0, speed * 6.375)

    # 计算G通道
    if only_r:
        result_image[..., 1] = np.zeros((row, col), dtype=np.uint8)
    else:
        result_image[..., 1] = ((np.cos(2 * np.pi * direction / 359) + 1) * 127.5).astype(np.uint8)

    # 计算B通道
    if only_r:
        result_image[..., 2] = np.zeros((row, col), dtype=np.uint8)
    else:
        result_image[..., 2] = ((np.sin(2 * np.pi * direction / 359) + 1) * 127.5).astype(np.uint8)

    SpeedThreshold = 14
    MaskImage = np.where(speed < SpeedThreshold, 0, 1)
    result_image[..., 0] = result_image[..., 0] * MaskImage
    result_image[..., 1] = result_image[..., 1] * MaskImage
    result_image[..., 2] = result_image[..., 2] * MaskImage

    # 创建一个PIL图像对象
    image = Image.fromarray(result_image)
    return image


# 产生图片
def generate_image(filepath, image_size):
    stacked_array, lon_range, lat_range = micaps_data_io.read_micaps_11(filepath)
    resized_array = cv2.resize(stacked_array, image_size)
    speed, direction = compute_speed_and_direction(resized_array)
    image = compute_rbg(speed, direction)
    return image


# 只产生R通道的图片（表示风速大小）
def generate_image_only_r(filepath, image_size):
    stacked_array, lon_range, lat_range = micaps_data_io.read_micaps_11(filepath)
    resized_array = cv2.resize(stacked_array, image_size)
    speed, direction = compute_speed_and_direction(resized_array)
    image = compute_rbg(speed, direction, only_r=True)
    return image


# image1 = generate_image('../data/20070908.000', (512, 320))
# image1.show()
