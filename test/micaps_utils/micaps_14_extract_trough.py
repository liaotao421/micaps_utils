# 从中央气象台的标注数据中，提取出槽线的标注
import re
import os


def file_change(src_file, target_file):
    with open(src_file, 'r') as file:
        lines = file.readlines()

    # 提取表示槽线的内容为 content_after_line
    start_index = None
    for i, line in enumerate(lines):
        if "WithProp_LINESYMBOLS" in line:
            start_index = i + 1
            break

    content_after_line = ''
    # 提取从"WithProp_LINESYMBOLS: 7"后面一行开始的内容
    if start_index is not None:
        content_after_line = ''.join(lines[start_index:])
    else:
        print("未找到指定行")


    pattern = re.compile(r'0 4 255 165 42 42 0 0(.*?)NoLabel 0', re.DOTALL)
    matches = pattern.findall(content_after_line)

    if len(matches) == 0:
        pattern = re.compile(r'0 4 255 255 255 0 0 0(.*?)NoLabel 0', re.DOTALL)
        matches = pattern.findall(content_after_line)

    data = []
    # 输出匹配到的内容
    for match in matches:
        content = match.strip()

        # 使用正则表达式提取点的个数
        point_count_match = re.search(r'(\d+)', content)
        if point_count_match:
            point_count = int(point_count_match.group())

        # 使用正则表达式提取经纬度
        data_matches = re.findall(r'\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', content)
        points = []
        if data_matches:
            for match in data_matches:
                longitude, latitude, other_value = map(float, match)
                point = (longitude, latitude)
                points.append(point)

        data.append(points)

    # data就是线条条数和经纬度的坐标

    # 写入到micaps14类数据
    file_name = os.path.basename(src_file)
    file = open(f'{target_file}', 'w')

    year, month, day = '20' + file_name[0:2], file_name[2:4], file_name[4:6]

    file.writelines('diamond 14\n')
    file.writelines(f'{year} {month} {day} 0 0\n')
    file.writelines('LINES: 0\n')

    # 线的个数
    file.writelines(f'LINES_SYMBOL: {len(data)}\n')

    for points in data:
        # points = points[::5]
        file.writelines(f'0 4 {len(points)}\n')
        index = 0
        for i in range(0, len(points), 1):
            index += 1
            x, y = points[i]
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
    file.writelines(f'WithProp_LINESYMBOLS: {len(data)}\n')

    for points in data:
        # points = points[::5]
        file.writelines(f'0 4 255 165 42 42 0 0\n')
        file.writelines(f'{len(points)}\n')
        index = 0
        for i in range(0, len(points), 1):
            index += 1
            x, y = points[i]
            file.write("{:10.3f}{:10.3f}     0.000".format(x, y))
            if index % 4 == 0:
                file.write('\n')
        if len(points) % 4 != 0:
            file.write('\n')
        file.write('NoLabel 0\n')

