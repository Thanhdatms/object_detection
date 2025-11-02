# import torch

# conf_scores_cl = torch.tensor([0.005, 0.2, 0.9, 0.008])  # 4 boxes
# conf_thresh = 0.01

# c_mask = conf_scores_cl.gt(conf_thresh)
# value = conf_scores_cl[c_mask]
# print(value)
# print(c_mask)

# import torch

# loss_conf = torch.tensor([3, 7, 4, 2])

# _, loss_idx = loss_conf.sort(dim=0, descending=True)
# _, idx_rank = loss_idx.sort(dim=0)

# print(loss_idx)
# print(idx_rank)
# print(loss_conf)


import torch
import torch.nn.functional as F

# # Giả sử:
# num_batch = 2        # 2 ảnh trong batch
# num_dbox = 3         # 3 default boxes mỗi ảnh
# num_classes = 4      # 4 lớp (0: background, 1-3: vật thể)

# # Dự đoán của model: [num_batch, num_dbox, num_classes]
# conf_data = torch.tensor([
#     [[2.0, 0.5, 0.3, 0.1],    # ảnh 1 - box 1
#      [0.1, 2.0, 0.2, 0.1],    # ảnh 1 - box 2
#      [0.5, 0.2, 1.5, 0.1]],   # ảnh 1 - box 3

#     [[1.0, 0.2, 0.3, 0.5],    # ảnh 2 - box 1
#      [0.3, 0.1, 1.0, 0.2],    # ảnh 2 - box 2
#      [0.2, 0.4, 0.1, 1.5]]    # ảnh 2 - box 3
# ])

# # Ground truth label cho từng default box
# conf_t_label = torch.tensor([
#     [1, 0, 2],  # ảnh 1: box1->class1, box2->background, box3->class2
#     [0, 2, 3]   # ảnh 2: box1->background, box2->class2, box3->class3
# ])

# # Flatten để tính loss
# batch_conf = conf_data.view(-1, num_classes)        # [6, 4]
# targets = conf_t_label.view(-1)                     # [6]

# # Tính loss
# loss = F.cross_entropy(batch_conf, targets, reduction='none')

# _, loss_idx = loss.sort(dim=0, descending=True)
# _, idx_rank = loss_idx.sort(dim=0)

# neg = idx_rank.unsqueeze(2)
# print(neg)

mask = torch.BoolTensor([1, 0, 1, 0])

data = torch.Tensor([
    [1,1,1,1],
    [2,2,2,2],
    [3,3,3,3],
    [4,4,4,4]
])

new_data = data[mask]
print(new_data)