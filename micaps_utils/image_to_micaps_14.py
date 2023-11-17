# 神经网络的输出转为micaps14类数据
import cv2
from skimage import morphology
import numpy as np
from sklearn.cluster import DBSCAN
import os


# 骨架提取
def skeleton(image):
    skeleton0 = morphology.skeletonize(image)
    return skeleton0.astype(np.uint8) * 255


# 得到图片中白色的点
def get_white_points(image):
    # 转换图像为灰度

    # 先来个反转
    gray_image = image[::-1, :]

    # 设置白色像素的阈值
    threshold = 144

    # 找到白色像素的坐标
    white_pixel_coordinates = np.argwhere(gray_image > threshold)

    return white_pixel_coordinates


# 对点做DBSCAN聚类
def dbscan_points(white_pixel_coordinates, eps=3, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)  # 调整eps和min_samples
    dbscan.fit(white_pixel_coordinates)

    # 获取每个点所属的聚类标签
    labels = dbscan.labels_

    # 创建一个字典，将点按照聚类标签分类

    clustered_points = {}

    for i, label in enumerate(labels):
        if label in clustered_points:
            clustered_points[label].append(white_pixel_coordinates[i])
        else:
            clustered_points[label] = [white_pixel_coordinates[i]]

    return clustered_points


def write_to_micaps14(clustered_points, file_path):
    file = open(file_path, 'w')


    file_name = os.path.basename(file_path)

    year, month, day = '20' + file_name[0:2], file_name[2:4], file_name[4:6]

    file.writelines('diamond 14\n')
    file.writelines(f'{year} {month} {day} 0 0\n')
    file.writelines('LINES: 0\n')

    # 线的个数
    file.writelines(f'LINES_SYMBOL: {len(clustered_points.items())}\n')

    for label, points in clustered_points.items():
        # points = points[::5]

        # 要对点就行排序
        x_y_points = []
        for point in points:
            x = point[0]
            y = point[1]
            p = (x, y)
            x_y_points.append(p)

        sorted_points = sorted(x_y_points, key=lambda x_y_points: (x_y_points[0], x_y_points[1]))

        file.writelines(f'0 4 {len(points)}\n')
        index = 0
        for i in range(0, len(sorted_points), 1):
            index += 1
            x, y = sorted_points[i]
            x = x / 3.2
            y = (68 * y + 3840) / 320
            file.write("{:10.3f}{:10.3f}     0.000".format(x, y))
            if index % 4 == 0:
                file.write('\n')
        if len(points) % 4 != 0:
            file.write('\n')
        file.write('NoLabel 0\n')

    file.writelines('SYMBOLS: 0\n')
    file.writelines('CLOSED_CONTOURS: 0\n')
    file.writelines('STATION_SITUATION\n')
    file.writelines('WEATHER_REGION: 0\n')
    file.writelines('FILLAREA: 0\n')
    file.writelines(f'WithProp_LINESYMBOLS: {len(clustered_points.items())}\n')

    for label, points in clustered_points.items():
        # points = points[::5]
        x_y_points = []
        for point in points:
            x = point[1]
            y = point[0]
            p = (x, y)
            x_y_points.append(p)

        sorted_points = sorted(x_y_points, key=lambda x_y_points: (x_y_points[0], x_y_points[1]))

        file.writelines(f'0 4 255 165 42 42 0 0\n')
        file.writelines(f'{len(points)}\n')
        index = 0
        for i in range(0, len(sorted_points), 1):
            index += 1
            x, y = sorted_points[i]
            x = x / 3.2
            y = (68 * y + 3840) / 320
            file.write("{:10.3f}{:10.3f}     0.000".format(x, y))
            if index % 4 == 0:
                file.write('\n')
        if len(points) % 4 != 0:
            file.write('\n')
        file.write('NoLabel 0\n')


def generate_micaps14_file(image_path, file_path):
    image = cv2.imread(image_path, 0)
    image = skeleton(image)
    white_pixel_coordinates = get_white_points(image)
    clustered_points = dbscan_points(white_pixel_coordinates)
    write_to_micaps14(clustered_points, file_path)

