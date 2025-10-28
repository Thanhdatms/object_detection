from make_data_path import make_datapath
from transform import DataTransform
from extract_inform import AnnotationExtractor
import cv2
import torch.utils.data as data
import torch
import numpy as np 


class MyDataSet(data.Dataset):
    def __init__(self, img_list, anno_list, phase, transform, anno_xml):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.anno_xml = anno_xml

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img, gt, height, width = self.pull_item(index)

        return img, gt

    def pull_item(self, index):
        img_file_path = self.img_list[index]
        img = cv2.imread(img_file_path) # BGR
        height, width, channels = img.shape

        # get annotation information
        anno_file_path = self.anno_list[index]
        anno_info = self.anno_xml(anno_file_path, width, height)

        # transform 
        img, boxes, labels = self.transform(img, self.phase, anno_info[:, :4], anno_info[:, 4])

        # convert BGR -> RGB
        # Numpy vs OpenCV ( H, W, C )
        # CNN and Pytorch ( C, H, W )
        img = torch.from_numpy(img[:, :, (2,1,0)]).permute(2,0,1)

        # setup ground truth
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, gt, height, width
    
def my_collate_fn(batch):   
    targets = []
    imgs = []

    for sample in batch:
        imgs.append(sample[0]) # add 1 more dimension at first
        targets.append(torch.FloatTensor(sample[1]))
    imgs = torch.stack(imgs, dim=0)

    return imgs, targets


if __name__ == "__main__":
    classes = ["aeroplane", "bicycle", "bird",  "boat", "bottle", 
               "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]

    # prepare train, valid, annotation list
    root_path = './datasets/VOC2012/'
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath(root_path)

    # prepare data transform
    color_mean = (104, 117, 123)
    input_size = 300

    train_dataset = MyDataSet(
        train_img_list, 
        train_annotation_list, 
        phase="train",
        transform=DataTransform(input_size, color_mean), 
        anno_xml=AnnotationExtractor(classes)
    )

    val_dataset = MyDataSet(
        val_img_list,
        val_annotation_list,
        phase="val",
        transform=DataTransform(input_size, color_mean),
        anno_xml=AnnotationExtractor(classes)
    )

    # print(len(train_dataset))
    # print(train_dataset.__getitem__(1))

    batch_size = 4
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)

    dataloader_dict = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    batch_iter = iter(dataloader_dict["val"])
    images, targets = next(batch_iter) # get 1 sample
    print(images.size()) 
    print(len(targets))
    print(targets[0].size()) # xmin, ymin, xmax, ymax, label
