import torch

conf_scores_cl = torch.tensor([0.005, 0.2, 0.9, 0.008])  # 4 boxes
conf_thresh = 0.01

c_mask = conf_scores_cl.gt(conf_thresh)
value = conf_scores_cl[c_mask]
print(value)
print(c_mask)