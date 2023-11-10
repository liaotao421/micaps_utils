from micaps_utils import micaps_11_to_image


image1 = micaps_11_to_image.generate_image('./sample_data/micaps_11/14050220.000', (512, 320))
image1.show()

image2 = micaps_11_to_image.generate_image_only_r('./sample_data/micaps_11/14050220.000', (512, 320))
image2.show()