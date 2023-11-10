# 神经网络的输出转为micaps14类数据
import cv2
from skimage import morphology
import numpy as np
from sklearn.cluster import DBSCAN
from PIL import Image


# 骨架提取
def skeleton(image):
    skeleton0 = morphology.skeletonize(image)
    return skeleton0.astype(np.uint8) * 255


# 得到图片中白色的点
def get_white_points(image):

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
    file.writelines('diamond 14\n')
    file.writelines('1989 06 04 0 0\n')
    file.writelines('LINES: 0\n')

    # 线的个数
    file.writelines(f'LINES_SYMBOL: {len(clustered_points.items())}\n')

    for label, points in clustered_points.items():
        # points = points[::5]
        file.writelines(f'0 4 {len(points)}\n')
        index = 0
        for i in range(0, len(points), 1):
            index += 1
            x = points[i][0]
            y = points[i][1]
            x = (68 * x + 3840) / 320
            y = y / 3.2
            file.write("{:10.3f}{:10.3f}     0.000".format(y, x))
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
        file.writelines(f'0 4 255 165 42 42 0 0\n')
        file.writelines(f'{len(points)}\n')
        index = 0
        for i in range(0, len(points), 1):
            index += 1
            x = points[i][0]
            y = points[i][1]
            x = (68 * x + 3840) / 320
            y = y / 3.2
            file.write("{:10.3f}{:10.3f}     0.000".format(y, x))
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


# test
# image = cv2.imread('../sample_data/network_output/19010508.png', 0)
# image = skeleton(image)
# image = Image.fromarray(image)
# image.show()