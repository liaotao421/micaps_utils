# micaps11类，4类，14类数据的读写操作
import numpy as np


def read_micaps_11(filepath):
    """
    :param filepath: micaps 11类数据的路径
    :return: stacked_array: 将u,v叠加后返回
             lon_range: 经度范围
             lat_range：维度范围
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # 1.前面几行提取有用信息
    # lon_interval, lat_interval, lon_range, lat_range, origin_size
    # 经度格距 纬度格距 经度范围 纬度范围 纬向格点数 经向格点数
    row_2 = [float(x) for x in lines[2].split()]
    lon_interval, lat_interval, lon_range, lat_range, origin_size = row_2[0], row_2[1], (row_2[2], row_2[3]), (row_2[4], row_2[5]), (row_2[6], row_2[7])

    # 2.找到换行行,可以读十行，找到哪一行的length最短
    new_line_flag = 100
    for line in lines[3:14]:
        line_float = [float(x) for x in line.split()]
        if len(line_float) < new_line_flag:
            new_line_flag = len(line_float)

    # 3.将数据读为（origin_size,2） 如 （33,21,2）
    data = []
    for line in lines[3:]:
        row = [float(x) for x in line.split()]
        data.append(row)

    result = []
    result_row = []
    for row in data:
        for element in row:
            result_row.append(element)
        if len(row) == new_line_flag:
            result.append(result_row)
            result_row = []

    np_array = np.array(result)

    split_arrays = np.split(np_array, 2)
    stacked_array = np.stack(split_arrays, axis=2)

    return stacked_array, lon_range, lat_range


def read_micaps_14(filepath):
    '''
    读取micaps第14类数据，micaps数据类型介绍如下链接：
    此代码来源：https://blog.csdn.net/weixin_43718675/article/details/89428532

    input: 文件路径
    output: micaps底图上叠加的线条对应的经纬度
    '''

    # 读取文件
    f = open(filepath, mode='r')
    data = f.readlines()

    # 获取该文件中包含几条线
    num_drylines = int(data[3].strip().split()[-1])

    n = 4  # 前五行为数据说明
    all_position = []

    # 获取每条线的经纬度坐标
    for i in range(num_drylines):

        # 获取该线条用多少个点来描述，其中每个点对应一个经度和维度(和一个额外的数字，不重要)
        n_points = int(data[n].strip().split()[-1])

        # 由于默认每行仅支持输出个4数据点，因此，会存在多行数据
        n_rows = int(np.ceil(n_points / 4))

        # 把包含这条线所有点的列连接起来
        point_data = data[n + 1]
        for i in range(n_rows - 1):
            point_data = point_data + data[n + 2 + i]


        # 去除换行符和将每个数据隔开
        point_data = point_data.strip().split()

        # 将数据类型由str转换为float型
        point_data = [float(point_data[i]) for i in range(len(point_data))]

        # 前面说到，每个点包含三个数据，其中第一个为经度，第二个为维度
        point_lon = point_data[0::3]
        point_lat = point_data[1::3]

        # 将该条线的经纬度坐标记录下来
        coordinate = [point_lon, point_lat]
        all_position.append(coordinate)

        # 每条线差两列
        n = n + n_rows + 2

    return all_position, data


def read_micaps_4(filepath):
    """
    :param filepath: micaps 4类数据的路径
    :return: np_array: 读到文件里的二维数组返回
             lon_range: 经度范围
             lat_range：维度范围
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # 1.前面几行提取有用信息
    # lon_interval, lat_interval, lon_range, lat_range, origin_size
    # 经度格距 纬度格距 经度范围 纬度范围 纬向格点数 经向格点数
    row_2 = [float(x) for x in lines[2].split()]
    lon_interval, lat_interval, lon_range, lat_range = row_2[0], row_2[1], (row_2[2], row_2[3]), (row_2[4], row_2[5])
    row_3 = [int(x) for x in lines[3].split()]
    origin_size = (row_3[0], row_3[1])

    # 2.找到换行行,可以读十行，找到哪一行的length最短
    new_line_flag = 100
    for line in lines[4:14]:
        line_float = [float(x) for x in line.split()]
        if len(line_float) < new_line_flag:
            new_line_flag = len(line_float)

    # 3.将数据读为（origin_size,2） 如 （33,21,2）
    data = []
    for line in lines[4:]:
        row = [float(x) for x in line.split()]
        data.append(row)

    result = []
    result_row = []
    for row in data:
        for element in row:
            result_row.append(element)
        if len(row) == new_line_flag:
            result.append(result_row)
            result_row = []

    np_array = np.array(result)
    return np_array, lon_range, lat_range

