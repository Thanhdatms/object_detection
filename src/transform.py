from make_data_path import make_datapath
from extract_inform import AnnotationExtractor
from utils.augmentation import Compose, ConvertFromInts, ToAbsoluteCoords, \
    PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, \
    ToPercentCoords, Resize, SubtractMeans

import cv2
import numpy as np
import matplotlib.pyplot as plt

class DataTransform:
    def __init__(self, input_size, color_mean):
        self.data_transform = {
            "train": Compose([
                ConvertFromInts(),      # convert image from int to float 32
                ToAbsoluteCoords(),     # back annotation to actual size
                PhotometricDistort(),   # change color by random
                Expand(color_mean),     
                RandomSampleCrop(),
                RandomMirror(),
                ToPercentCoords(),       # convert annotation to [0,1]
                Resize(input_size), # covert image to input size 300
                SubtractMeans(color_mean)# related to color BGR 
            ]),

            # "val": Compose([
            #     ConvertFromInts(),
            #     ToPercentCoords(),
            #     Resize(size=input_size),
            #     SubtractMeans(mean=color_mean)
            # ])
            "val": Compose([
                ConvertFromInts(),
                ToAbsoluteCoords(),
                Resize(input_size),
                ToPercentCoords(),
                SubtractMeans(color_mean)
            ])
        }

    def __call__(self, img, phase, boxes, labels):
        return self.data_transform[phase](img, boxes, labels)

    
if __name__ == "__main__":
    classes = ["aeroplane", "bicycle", "bird",  "boat", "bottle", 
               "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]
    
    root_path = './datasets/VOC2012/'
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath(root_path=root_path)
    print(train_img_list[0])
    idx = 0
    img = cv2.imread(train_img_list[0], cv2.COLOR_BGR2RGB)
    height, width, channels = img.shape

    # prepare data transform
    annotation = AnnotationExtractor(class_names=classes)
    annotation_img_list = annotation(train_annotation_list[idx], width, height)
    color_mean = (104, 117, 123)
    input_size = 300
    data_transform = DataTransform(input_size=input_size, color_mean=color_mean)

    # train phase
    train_img_transformed, train_boxes, train_labels = data_transform(img, 'train', annotation_img_list[:, :4], annotation_img_list[:, 4])
    h, w, _ = train_img_transformed.shape
    train_boxes_pixel = train_boxes.copy()
    train_boxes_pixel[:, [0,2]] *= w
    train_boxes_pixel[:, [1,3]] *= h
    train_boxes_pixel = train_boxes_pixel.astype(np.int32)

    # val phase
    val_img_transformed, val_boxes, val_labels = data_transform(img, 'val', annotation_img_list[:, :4], annotation_img_list[:, 4])
    h, w, _ = val_img_transformed.shape
    val_boxes_pixel = val_boxes.copy()
    val_boxes_pixel[:, [0,2]] *= w
    val_boxes_pixel[:, [1,3]] *= h
    val_boxes_pixel = val_boxes_pixel.astype(np.int32)

    def draw_boxes(image, boxes, labels, title):
        img_copy = image.copy()
        for box, label_idx in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            color = (0, 255, 0)
            cv2.rectangle(img_copy, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(img_copy, classes[int(label_idx)], (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        
    # Plot img
    plt.figure(figsize=(12,8))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original")

    plt.subplot(1, 3, 2)
    plt.imshow(draw_boxes(val_img_transformed.copy(), val_boxes_pixel, val_labels, "Val"))
    plt.title("valdidation")

    plt.subplot(1, 3, 3)
    plt.imshow(draw_boxes(train_img_transformed.copy(), train_boxes_pixel, train_labels, "Train"))
    plt.title("Training")

    
    plt.tight_layout
    plt.show()