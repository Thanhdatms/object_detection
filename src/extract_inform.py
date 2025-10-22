from make_data_path import make_datapath
import xml.etree.ElementTree as ET
import numpy as np
import cv2

class AnnotationExtractor:
    def __init__(self, class_names: list):
        self.class_names = class_names

    def __call__(self, xml_path, width, height):
        
        ret = []

        # real xml file
        xml = ET.parse(xml_path).getroot()

        for obj in xml.iter('object'):
            # ignore difficult label
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue

            # get bounding box information
            bndbox = []
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']

            for pt in pts:
                pixel = int(bbox.find(pt).text) - 1
                if pt == 'xmin' or pt == 'ymin':
                    pixel /= width
                else:
                    pixel /= height

                bndbox.append(pixel)
            
            label_id = self.class_names.index(name)
            bndbox.append(label_id)

            ret += [bndbox]

        return np.array(ret)
    

if __name__ == "__main__":

    root_path = './datasets/pascal-voc-2012/VOC2012_train_val/VOC2012_train_val/'

    classes = ["aeroplane", "bicycle", "bird",  "boat", "bottle", 
               "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]
    
    annotation = AnnotationExtractor(class_names=classes)
    idx = 1

    train_image_list, train_annotation_list, val_image_list, val_annotation_list = make_datapath(root_path=root_path)
    img = cv2.imread(train_image_list[idx])
    height, width, channels = img.shape

    annotation_data_test = annotation(train_annotation_list[idx], 300, 300)
    print(annotation_data_test)

