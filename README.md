# micaps_utils
读micap4,11,14类数据。并转为神经网络训练的图像

## micaps4 to image

读取micaps4类数据，并按论文中方法处理为指定大小的图片

论文中方法：

![image-20231115152140350](https://edu-cubeonline.oss-cn-chengdu.aliyuncs.com/image-20231115152140350.png)

实现效果:

```python
from micaps_utils import micaps_4_to_image

image1 = micaps_4_to_image.generate_image('./sample_data/micaps_4/19020108.000', (512, 320))
image1.show()
```

![image-20231115152321683](https://edu-cubeonline.oss-cn-chengdu.aliyuncs.com/image-20231115152321683.png)

## micaps11 to image

读取micaps11类数据，并按论文中方法处理为指定大小的图片

论文中方法：

大概是根据u,v两个分量计算分向和风速，然后按下面的公式计算RBG三个通道的值

![image-20231115152655491](https://edu-cubeonline.oss-cn-chengdu.aliyuncs.com/image-20231115152655491.png)

实现效果：

```python
from micaps_utils import micaps_11_to_image


image1 = micaps_11_to_image.generate_image('./sample_data/micaps_11/14050220.000', (512, 320))
image1.show()

# 只有红色通道的图片是后面可能有用
image2 = micaps_11_to_image.generate_image_only_r('./sample_data/micaps_11/14050220.000', (512, 320))
image2.show()
```

![image-20231115152840265](https://edu-cubeonline.oss-cn-chengdu.aliyuncs.com/image-20231115152840265.png)

![image-20231115152850119](https://edu-cubeonline.oss-cn-chengdu.aliyuncs.com/image-20231115152850119.png)

## micaps14 to image

读取micaps14类数据，并按论文中方法处理为指定大小的图片

这里将论文中计算点到每条线段的距离修改为直接计算点到点的距离，并对只有两个点的情况特殊处理

实现效果：

```
from micaps_utils import micaps_14_to_image

# 处理一张图片大约需要30s
image1 = micaps_14_to_image.generate_image_new('./sample_data/micaps_14/19020108', (320, 512), 5)
image1.show()
```

![](https://edu-cubeonline.oss-cn-chengdu.aliyuncs.com/image-20231115153313610.png)

## get_trough_line

下载气象台的标注结果有槽线以外的其他标注，需要只提取出标注的槽线

![image-20231115153709356](https://edu-cubeonline.oss-cn-chengdu.aliyuncs.com/image-20231115153709356.png)

实现效果：

```python
from micaps_utils.micaps_14_extract_trough import file_change

src_file = './sample_data/qixiangtai_micaps14/21011508.000'
target_file = './sample_data/qixiangtai_micaps14/21011508'

file_change(src_file, target_file)
```

![image-20231115153926543](https://edu-cubeonline.oss-cn-chengdu.aliyuncs.com/image-20231115153926543.png)

## image to micaps14

将神经网络的输出进行骨架提取，聚类后，写为micaps14类数据

实现效果：

```python
from micaps_utils import image_to_micaps_14

file_path = './19010508'

image_to_micaps_14.generate_micaps14_file('./sample_data/network_output/19010508.png', file_path=file_path)
```

![image-20231115161001269](https://edu-cubeonline.oss-cn-chengdu.aliyuncs.com/image-20231115161001269.png)

左边是神经网络的输出，右边是转为micaps14类数据后的结果

