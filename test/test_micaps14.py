from micaps_utils import micaps_14_to_image


image1 = micaps_14_to_image.generate_image_new('./sample_data/micaps_14/19020108', (320, 512), 5)
image1.show()
