import torch.nn as nn
from l2norm import L2Norm
from  default_box import DefaultBox


def vgg():
    layers = []
    in_channels = 3

    cfgs = [64, 64, 'M', 
           128, 128, 'M',
           256, 256, 256, 'M',
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
    "input_szie": 300,
    "bbox_aspect_num": [4, 6, 6, 6, 4, 4],
    "feature_maps": [38, 19, 10, 5, 3, 1],
    "steps": [8, 16, 32, 64, 100, 300],

    "min_sizes": [30, 60, 111, 162, 213, 264], # min_sizes for default boxes
    "max_sizes": [60, 111, 162, 213, 264, 315], # max_sizes for default boxes
    "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]], # aspect ratios for default boxes
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

if __name__ == "__main__":
    # layers = create_vgg()
    # print(layers)

    # extra_layers = extras()
    # print(extra_layers)
     
    # locate_layers, confidence_layers = locate_confidence()
    # print(locate_layers, confidence_layers)

    ssd = SSD(phase='train', cfg=cfgs)

    print(ssd)

    
    
