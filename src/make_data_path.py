import os.path as osp


def make_datapath(root_path):
    img_path_template = osp.join(root_path, 'JPEGImages', '%s.jpg')
    annotation_path_template = osp.join(root_path, 'Annotations', '%s.xml')

    train_id_names = osp.join(root_path, 'ImageSets/Main/train.txt')
    val_id_names = osp.join(root_path, 'ImageSets/Main/val.txt')

    train_img_list = []
    train_anno_list = []
    val_img_list = []
    val_anno_list = []

    for train in open(train_id_names):
        train_id = train.strip()

        img_path = (img_path_template % train_id)
        anno_path = (annotation_path_template % train_id)

        train_img_list.append(img_path)
        train_anno_list.append(anno_path)


    for val in open(val_id_names):
        val_id = train.strip()

        img_path = (img_path_template % val_id)
        anno_path = (annotation_path_template % val_id)

        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list


if __name__ == "__main__":
    root_path = './datasets/pascal-voc-2012/VOC2012_train_val/VOC2012_train_val/'

    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath(root_path)
    print(len(train_img_list))