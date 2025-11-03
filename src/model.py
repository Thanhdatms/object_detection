import torch.nn as nn
import torch
from l2norm import L2Norm
from  default_box import DefaultBox
from torch.autograd import Function

def vgg():
    layers = []
    in_channels = 3

    cfgs = [64, 64, 'M', 
           128, 128, 'M',
           256, 256, 256, 'MC',
           512, 512, 512, 'M',
           512, 512, 512
        ]
    # Max Pooling layer
    # M: kernel_size=2, stride=2, ceil_mode=False, chi lay phan nguyen
    # MC: kernel_size=2, stride=2, ceil_mode=True, lam tron len

    for cfg in cfgs:
        if cfg == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif cfg == 'MC':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, cfg, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = cfg

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)

    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    
    return nn.ModuleList(layers)

def extras():
    layers = []
    in_channels = 1024

    cfgs = [256, 512, 128, 256, 128, 256, 128, 256]

    layers += [nn.Conv2d(in_channels, cfgs[0], kernel_size=1)]
    layers += [nn.Conv2d(cfgs[0], cfgs[1], kernel_size=3, stride=2, padding=1)]

    layers += [nn.Conv2d(cfgs[1], cfgs[2], kernel_size=1)]
    layers += [nn.Conv2d(cfgs[2], cfgs[3], kernel_size=3, stride=2, padding=1)]

    layers += [nn.Conv2d(cfgs[3], cfgs[4], kernel_size=1)]
    layers += [nn.Conv2d(cfgs[4], cfgs[5], kernel_size=3)]

    layers += [nn.Conv2d(cfgs[5], cfgs[6], kernel_size=1)]
    layers += [nn.Conv2d(cfgs[6], cfgs[7], kernel_size=3)]

    return nn.ModuleList(layers)

def locate_confidence(classes=21, bbox_aspect_ratios = [4, 6, 6, 6, 4, 4]):
    locate_layers = []
    confidence_layers = []

    # source 1: 512
    locate_layers += [nn.Conv2d(512, bbox_aspect_ratios[0] * 4, kernel_size=3, padding=1)]
    confidence_layers += [nn.Conv2d(512, bbox_aspect_ratios[0] * classes, kernel_size=3, padding=1)]

    # source 2: 1024
    locate_layers += [nn.Conv2d(1024, bbox_aspect_ratios[1] * 4, kernel_size=3, padding=1)]
    confidence_layers += [nn.Conv2d(1024, bbox_aspect_ratios[1]*classes, kernel_size=3, padding=1)]

    # source 3: 512
    locate_layers += [nn.Conv2d(512, bbox_aspect_ratios[2] * 4, kernel_size=3, padding=1)]
    confidence_layers += [nn.Conv2d(512, bbox_aspect_ratios[2]*classes, kernel_size=3, padding=1)]

    # source 4: 256
    locate_layers += [nn.Conv2d(256, bbox_aspect_ratios[3] * 4, kernel_size=3, padding=1)]
    confidence_layers += [nn.Conv2d(256, bbox_aspect_ratios[3]*classes, kernel_size=3, padding=1)]

    # source 5: 256
    locate_layers += [nn.Conv2d(256, bbox_aspect_ratios[4] * 4, kernel_size=3, padding=1)]
    confidence_layers += [nn.Conv2d(256, bbox_aspect_ratios[4]*classes, kernel_size=3, padding=1)]

    # source 6: 256
    locate_layers += [nn.Conv2d(256, bbox_aspect_ratios[5] * 4, kernel_size=3, padding=1)]
    confidence_layers += [nn.Conv2d(256, bbox_aspect_ratios[5]*classes, kernel_size=3, padding=1)]

    return nn.ModuleList(locate_layers), nn.ModuleList(confidence_layers)


cfgs = {
    "num_classes": 21,
    "input_size": 300,
    "bbox_aspect_num": [4, 6, 6, 6, 4, 4],
    "feature_maps": [38, 19, 10, 5, 3, 1],
    "steps": [8, 16, 32, 64, 100, 300],
    "min_sizes": [30, 60, 111, 162, 213, 264],
    "max_sizes": [60, 111, 162, 213, 264, 315],
    "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
}


class SSD(nn.Module):
    def __init__(self, phase, cfg):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = cfg['num_classes']

        # create main module
        self.vgg = vgg()
        self.extras = extras()
        self.locate, self.confidence = locate_confidence(classes=self.num_classes, bbox_aspect_ratios=cfg['bbox_aspect_num'])
        self.l2norm = L2Norm()

        # create default box
        dboxes = DefaultBox(cfgs=cfgs)
        self.dbox_list = dboxes.create_default_box()

        if phase == 'inference':
            self.detect = Detect()

    def forward(self, x):
        sources = []
        locate = []
        confidence = []

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        source_1 = self.l2norm(x)
        sources.append(source_1)

        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        
        # append source_2
        sources.append(x)

        for k in range(0, len(self.extras), 2):
            # apply source 3, 4, 5, 6
            x = self.extras[k](x)
            x = nn.ReLU(inplace=True)(x)
            x = self.extras[k+1](x)
            x = nn.ReLU(inplace=True)(x)
            sources.append(x)

        # apply locate and confidence head to source layers
        for (src, loc, conf) in zip(sources, self.locate, self.confidence):
            # aspect ratio num is 4, 6, 6, 6, 4, 4
            locate.append(loc(src).permute(0, 2, 3, 1).contiguous())
            confidence.append(conf(src).permute(0, 2, 3, 1).contiguous())
        
        # Neu khong co contiguous thi se bi loi khi view
        locate = torch.cat([loc.view(loc.size(0), -1) for loc in locate], dim=1) # (batch_size, 8732*4)
        confidence = torch.cat([conf.view(conf.size(0), -1) for conf in confidence], dim=1)

        locate = locate.view(locate.size(0), -1, 4) # batch_size, 8732, 4
        confidence = confidence.view(confidence.size(0),-1, self.num_classes) # batch_size, 8732, num_classes

        output = (locate, confidence, self.dbox_list)

        if self.phase ==  "inference": # during inference, apply detect function to locate, confidence and dbox_list
            with torch.no_grad():
                return self.detect(output[0], output[1], output[2]) # locate, confidence, dbox_list
        else:
            return output # during training, return locate, confidence, dbox_list


def decode(loc, defbox_list):
    """
    parameters:
    loc: [8732, 4] (delta_x, delta_y, delta_w, delta_h)
    defbox_list: [8732, 4] (cx_d, cy_d, w_d, h_d)

    returns:
    boxes [xmin, ymin, xmax, ymax]
    """

    boxes = torch.cat((
        defbox_list[:, :2] + 0.1*loc[:, :2]*defbox_list[:, 2:],
        defbox_list[:, 2:]*torch.exp(loc[:,2:]*0.2)), dim=1)

    boxes[:, :2] -= boxes[:,2:]/2 #calculate xmin, ymin
    boxes[:, 2:] += boxes[:, :2] #calculate xmax, ymax

    return boxes    

def nms(boxes, scores, overlap=0.5, top_k=200):
    """
    boxes: the output boxes after decode [num_box, xmin, ymin, xmax, ymax]
    scores: confidence score for each box [num_box]
    overlap: threshold for nms 
    top_k: maximum number of boxs to consider
    returns:
    """
    count = 0
    keep = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # calculate area of boxes
    area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    # sort scores and keep top_k highest score box
    _, order = scores.sort(0) # sort in ascending order, sort(0): ascending, sort(1): descending
    print("order:", order)
    order = order[-top_k:] # select top_k highest score box

    while order.numel() > 0:
        idx = order[-1] # index of current highest score box
        keep.append(idx.item())
        count += 1

        if order.numel() == 1:
            break
        order = order[:-1] # remove kept index from order

        # intersection box
        xx1 = torch.max(x1[idx], x1[order])
        yy1 = torch.max(y1[idx], y1[order])
        xx2 = torch.min(x2[idx], x2[order])
        yy2 = torch.min(y2[idx], y2[order])

        # clamp to minimum 0
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)

        inter = w * h
        union = area[idx] + area[order] - inter

        iou = inter / union

        # keep boxes with overlap less than threshold
        order = order[iou <= overlap]
    
    return torch.tensor(keep, dtype=torch.long), count


class Detect(Function):
    def __init__(self, conf_thresh=0.01, nms_thresh=0.5, top_k=200):
        self.soft_max = nn.Softmax(dim=-1)
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.top_k = top_k

    # loc_data: (batch_size, 8732, 4) list location vector of each box (delta_x, delta_y, delta_w, delta_h)
    # conf_data: (batch_size, 8732, num_classes) list label vector of each box
    # dbox_list: (8732, 4) list default box
    def forward(self, loc_data, conf_data, dbox_list):
        # 
        num_batch = loc_data.size(0) # batch size, how many image in a batch
        num_dbox = loc_data.size(1) # 8732 box
        num_classes = conf_data.size(2) # 21 classes

        conf_data = self.soft_max(conf_data) # apply softmax to confidence score like [batch_size, 8732, num_classes]
        # (batch_size, 8732, 21) -> (batch_size, num_classes, num_box)
        conf_preds = conf_data.transpose(2, 1)

        output = torch.zeros(num_batch, num_classes, self.top_k, 5) # (batch_size, num_classes, top_k, 5)
        # Bao nhiêu ảnh trong 1 batch -> mỗi ảnh gồm bao nhiêu class -> mỗi class sẽ có top k box -> mỗi box gồm (score, xmin, ymin, xmax, ymax)
        for i in range(num_batch):
            # decode bbox from offset and default box
            decoded_boxes = decode(loc_data[i], dbox_list)

            conf_scores = conf_preds[i] # (num_classes, num_box)
            for cls in range(1, num_classes):

                # create mask to get score for confidence score greater than threshold
                score_mask = conf_scores[cls].gt(self.conf_thresh)
                scores = conf_scores[cls][score_mask]
                if scores.numel() == 0:
                    continue
            
                # create mask to get location box for confidence score greater than threshold
                box_mask = score_mask.unsqueeze(1).expand_as(decoded_boxes) # (8732, 4)
                boxes = decoded_boxes[box_mask].view(-1, 4)

                # apply nms
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)

                output[i, cls, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), dim=1)

        return output


if __name__ == "__main__":
    # layers = create_vgg()
    # print(layers)

    # extra_layers = extras()
    # print(extra_layers)
     
    # locate_layers, confidence_layers = locate_confidence()
    # print(locate_layers, confidence_layers)

    # ssd = SSD(phase='train', cfg=cfgs)
    # print(ssd)

    boxes = torch.tensor([  [12, 12, 22, 22],
                            [30, 30, 40, 40],
                            [10, 10, 20, 20],
                            [10, 10, 20, 20],
                        ])
    scores = torch.tensor([0.9, 0.85, 0.8, 0.8])

    keep, count = nms(boxes, scores, overlap=0.5, top_k=200)
    print(keep, count)

    
    
