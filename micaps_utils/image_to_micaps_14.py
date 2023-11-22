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
    global new_line, temp_points
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)  # 调整eps和min_samples
    dbscan.fit(white_pixel_coordinates)

    # 获取每个点所属的聚类标签
    labels = dbscan.labels_

    # 创建一个字典，将点按照聚类标签分类
    clustered_points = {}
    pop_flag = 0
    for i, label in enumerate(labels):
        if label in clustered_points:
            clustered_points[label].append(white_pixel_coordinates[i])
        else:
            clustered_points[label] = [white_pixel_coordinates[i]]

    # 聚类后若有分叉要分为两条线
    # 找到有三个端点的线
    for label, pointss in clustered_points.items():
        points = []
        for point in pointss:
            x = point[0]
            y = point[1]
            p = (x, y)
            points.append(p)

        eight_neighbor_3 = []
        start_point = (0, 0)
        for point in points:
            x, y = point
            neighbor_num = 0
            eight_neighbor = [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), (x - 1, y), (x + 1, y), (x - 1, y + 1),
                              (x, y + 1), (x + 1, y + 1)]
            for p in points:
                if p in eight_neighbor:
                    neighbor_num += 1
            if neighbor_num == 4:
                start_point = point
            if neighbor_num == 3:
                eight_neighbor_3.append(point)

        if len(eight_neighbor_3) == 0:
            continue

        # 找到下一个点
        x, y = start_point
        eight_neighbor = [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), (x - 1, y), (x + 1, y), (x - 1, y + 1),
                          (x, y + 1),
                          (x + 1, y + 1)]
        next_point = (0, 0)
        for point in points:
            if point in eight_neighbor and point not in eight_neighbor_3:
                next_point = point

        # 另起一条线
        new_line = [start_point, next_point]
        points.remove(start_point)
        points.remove(next_point)

        while True:
            x, y = next_point
            neighbor_num = 0
            eight_neighbor = [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), (x - 1, y), (x + 1, y), (x - 1, y + 1),
                              (x, y + 1),
                              (x + 1, y + 1)]
            for p in points:
                if p in eight_neighbor:
                    neighbor_num += 1
            # 找到终点了
            if neighbor_num == 0:
                break
            for point in points:
                if point in eight_neighbor:
                    new_line.append(point)
                    next_point = point
                    points.remove(point)

        #
        pop_flag = label
        temp_points = points
    clustered_points.pop(pop_flag)

    clustered_points[67373] = [list(t) for t in new_line]
    clustered_points[67374] = [list(t) for t in temp_points]

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

        # 排序不行，还是得连通域这方向考虑

        points_num = len(x_y_points)

        start_point = ()

        for point in x_y_points:
            x, y = point
            neighbor_num = 0
            eight_neighbor = [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), (x - 1, y), (x + 1, y), (x - 1, y + 1),
                              (x, y + 1), (x + 1, y + 1)]
            for p in x_y_points:
                if p in eight_neighbor:
                    neighbor_num += 1
            if neighbor_num == 1:
                start_point = point

        # 找到起点/终点
        points_sorted = []
        points_sorted.append(start_point)

        x_y_points.remove(start_point)

        while len(points_sorted) < points_num:

            x, y = start_point
            eight_neighbor = [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), (x - 1, y), (x + 1, y), (x - 1, y + 1),
                              (x, y + 1),
                              (x + 1, y + 1)]

            for point in x_y_points:
                if point in eight_neighbor:
                    points_sorted.append(point)
                    start_point = point
                    x_y_points.remove(point)




        file.writelines(f'0 4 {len(points)}\n')
        index = 0
        for i in range(0, len(points_sorted), 1):
            index += 1
            x, y = points_sorted[i]
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

        # 排序不行，还是得连通域这方向考虑

        points_num = len(x_y_points)

        start_point = ()

        for point in x_y_points:
            x, y = point
            neighbor_num = 0
            eight_neighbor = [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), (x - 1, y), (x + 1, y), (x - 1, y + 1),
                              (x, y + 1), (x + 1, y + 1)]
            for p in x_y_points:
                if p in eight_neighbor:
                    neighbor_num += 1
            if neighbor_num == 1:
                start_point = point

        # 找到起点/终点
        points_sorted = []
        points_sorted.append(start_point)

        x_y_points.remove(start_point)

        while len(points_sorted) < points_num:

            x, y = start_point
            eight_neighbor = [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), (x - 1, y), (x + 1, y), (x - 1, y + 1),
                              (x, y + 1),
                              (x + 1, y + 1)]

            for point in x_y_points:
                if point in eight_neighbor:
                    points_sorted.append(point)
                    start_point = point
                    x_y_points.remove(point)

        file.writelines(f'0 4 255 165 42 42 0 0\n')
        file.writelines(f'{len(points)}\n')
        index = 0
        for i in range(0, len(points_sorted), 1):
            index += 1
            x, y = points_sorted[i]
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

