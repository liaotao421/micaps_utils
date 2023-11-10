# micaps14类数据变化为数据集的label

import numpy as np
from PIL import Image
from .micaps_data_io import read_micaps_14
import math


# 按照论文中的方法计算点到直线的距离
def dis(p1, p2, p3):
    '''
    :param p1: p1(x,y)
    :param p2: p2(x_1,,y_1)
    :param p3: p3(x_2,,y_2)
    :return: p1 到 p2和p3确定的直线的距离
    '''
    x, y = p1
    x_1, y_1 = p2
    x_2, y_2 = p3

    r = (x - x_2) * (x_1 - x_2) + (y - y_2) * (y_1 - y_2) / (x_1 - x_2) ** 2 + (y_1 - y_2) ** 2

    if 0 < r < 1:
        return math.sqrt((x_1 - x) ** 2 + (y_1 - y) ** 2 - (((x - x_2) * (x_1 - x_2) + (y - y_2) * (y_1 - y_2)) ** 2) / ((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2))
    elif r <= 0:
        return math.sqrt((x_1 - x) ** 2 + (y_1 - y) ** 2)
    elif r >= 1:
        return math.sqrt((x_2 - x) ** 2 + (y_2 - y) ** 2)


# 计算两点之间的距离
def dis1(p1 ,p2):
    x_1, y_1 = p1
    x_2, y_2 = p2
    return math.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)


# 按照论文中的方法生成图片(速度较慢)
def generate_image(filepath, image_size, threshold):
    '''
    :param filepath: micaps_14类文件路径
    :param image_size: 图片尺寸，如(320*512)
    :param threshold:  每个点距离标注的槽线的欧式距离的阈值，详见论文
    :return: 生成的图片
    '''

    coor_data, data = read_micaps_14(filepath)
    zero_array = np.zeros(image_size)
    row, col = zero_array.shape

    for i in range(col):
        for j in range(row):
            p1 = (i, j)
            for line in coor_data:
                jd = line[0]
                wd = line[1]
                for k in range(len(jd) - 1):
                    # 每个点的经纬度
                    # 经纬度坐标转化为格点坐标
                    p2 = jd[k] * 3.2, ((wd[k] - 12)/68) * 320
                    p3 = jd[k + 1] * 3.2, ((wd[k+1] - 12)/68) * 320
                    # 这里跳过的话会导致部分点截断
                    try:
                        distance = dis(p1, p2, p3)
                    except Exception as e:
                        continue
                    if distance <= threshold:
                        zero_array[j, i] = 255
                        break

    zero_array = zero_array[::-1, :]
    image = Image.fromarray(zero_array)
    return image


# 直接计算点的距离试试（速度稍快）
def generate_image_new(filepath, image_size, threshold):

    '''
    :param filepath: micaps_14类文件路径
    :param image_size: 图片尺寸，如(320*512)
    :param threshold:  每个点距离标注的槽线的欧式距离的阈值，详见论文
    :return: 生成的图片 image类型
    '''

    coor_data, data = read_micaps_14(filepath)
    zero_array = np.zeros(image_size)
    row, col = zero_array.shape

    for i in range(col):
        for j in range(row):
            p1 = (i, j)
            for line in coor_data:
                jd = line[0]
                wd = line[1]
                # 只有两个点的连线特殊情况
                if len(jd) == 2:
                    x1, y1 = jd[0], wd[0]
                    x2, y2 = jd[1], wd[1]
                    # 线性插值补充一些点
                    for i in range(1, 10, 1):
                        factor = i / 10
                        interpolated_x = x1 + factor * (x2 - x1)
                        interpolated_y = y1 + factor * (y2 - y1)
                        jd.append(interpolated_x)
                        wd.append(interpolated_y)
                for k in range(len(jd)):
                    # 每个点的经纬度
                    # 经纬度坐标转化为格点坐标
                    p2 = jd[k] * 3.2, ((wd[k] - 12) / 68) * 320
                    distance = dis1(p1, p2)

                    if distance <= threshold:
                        zero_array[j, i] = 255
                        break

    zero_array = zero_array[::-1, :]
    image = Image.fromarray(zero_array)
    return image


# demo
# micaps_file = ''
# image = generate_image_new(micaps_file, (320, 512), 6.4)








