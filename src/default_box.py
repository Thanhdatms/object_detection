from math import sqrt
import torch
import pandas as pd


# config for test in main
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


class DefaultBox():
    def __init__(self, cfgs):
        self.img_size = cfgs["input_szie"]
        self.feature_maps = cfgs["feature_maps"]
        self.min_sizes = cfgs["min_sizes"]
        self.max_sizes = cfgs["max_sizes"]
        self.aspect_ratios = cfgs["aspect_ratios"]
        self.steps = cfgs["steps"]

    def create_default_box(self):
        default_boxes = []
        for k, f in enumerate(self.feature_maps):
            for i in range(f):
                for j in range(f):
                    f_k = self.img_size / self.steps[k]
                    cx = (i + 0.5) / f_k
                    cy = (j + 0.5) / f_k
                    # small box
                    s_k = self.min_sizes[k] / self.img_size
                    default_boxes += [cx, cy, s_k, s_k]

                    # big box
                    s_k_prime = sqrt(s_k*(self.max_sizes[k] / self.img_size))
                    default_boxes += [cx, cy, s_k_prime, s_k_prime]

                    for ar in self.aspect_ratios[k]:
                        default_boxes += [cx, cy, s_k*sqrt(ar), s_k*sqrt(ar)]
                        default_boxes += [cx, cy, s_k / sqrt(ar), s_k / sqrt(ar)]

        output = torch.Tensor(default_boxes).view(-1, 4)
        output.clamp_(max=1, min=0)

        return output

if __name__ == "__main__":
    defboxes = DefaultBox(cfgs=cfgs)
    box_list = defboxes.create_default_box()
    print(pd.DataFrame(box_list.numpy()))



                     


