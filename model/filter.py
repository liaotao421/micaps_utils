from add_perturbation import filter_pseudo_labels

image = './jet_stream_dataset/image/18123120.png'



is_good, iou = filter_pseudo_labels(image, model_path='./model.pth', threshold=0.7)


print(iou)

