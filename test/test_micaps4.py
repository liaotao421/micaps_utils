from micaps_utils import micaps_4_to_image


image1 = micaps_4_to_image.generate_image('./sample_data/micaps_4/19020108.000', (512, 320))
image1.show()