# Jaccard
# Hard negative mining: negative default to 3x positive
# MSE -> Smooth L1 F.SmoothL1Loss
# Loss in classification (multi class) -> Categorical Cross Entropy F.CrossEntropy
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.boxes_utils import jaccard, match

class MultiBoxLoss(nn.Module):
    def __init__(self, jaccard_threshold=0.5, neg_pos=3, device='cpu'):
        super(MultiBoxLoss, self).__init__()
        self.jaccard_threshold = jaccard_threshold
        self.neg_pos = neg_pos
        self.device = device

    # def forward(self, predictions, targets):
    #     loc_data, conf_data, dbox_list = predictions
    #     # loc_data: tensor dự đoán vị trí (offsets) kích thước (batch, num_dbox, 4)
    #     # conf_data: tensor dự đoán class label (logits) kích thước (batch, num_dbox, num_classes)
    #     # dbox_list: tensor các default box kích thước (num_dbox, 4)

    #     # shape
    #     batch_size = loc_data.size(0)
    #     num_priors = loc_data.size(1)
    #     num_classes = conf_data.size(2)

    #     # init targets
    #     conf_t = torch.zeros((batch_size, num_priors), dtype=torch.long, device=self.device) # shape (batch_size, num_priors)
    #     loc_t = torch.zeros((batch_size, num_priors, 4), dtype=torch.float, device=self.device) # shape (batch_size, num_priors, 4)

    #     dbox = dbox_list.to(self.device)
    #     variances = [0.1, 0.2]

    #     for idx in range(batch_size):
    #         # get ground truth location and babel 
    #         truths = targets[idx][:, :-1]
    #         labels = targets[idx][:, -1]

    #         # create targets location regression and target classification
    #         match(
    #             threshold=self.jaccard_threshold,
    #             truths=truths,
    #             priors=dbox,
    #             variances=variances,
    #             labels=labels,
    #             loc_t=loc_t,
    #             conf_t=conf_t,
    #             idx=idx
    #             )
    #         # return
    #         # Giả sử num_priors = 8732
    #         # Từ conf_t và loc_t rỗng được khởi tạo trước đó bh đã có giá trị từ việc so khớp ground truth với bounding boxes
    #         # loc_t ([batch, 8732, 4]) (Δcx, Δcy, Δw, Δh) offset
    #         # conf_t ([batch, 8732]) label

    #     pos_mask = conf_t > 0 # [True, False, ....]

    #     # --------------------------------------
    #     # Calculate Localization loss (SmoothL1)
    #     # --------------------------------------
    #     pos_idx = pos_mask.unsqueeze(2).expand_as(loc_t) # mask for only positive priors

    #     loc_p = loc_data[pos_idx].contiguous().view(-1, 4) # lọc lại kết quả sau khi dự đoán
    #     loc_t_pos = loc_t[pos_idx].contiguous().view(-1, 4) # lọc lại kết quả groud truth

    #     # If there is no positive, loc_p and loc_t_pos can be empty tensors.
    #     if loc_p.numel() > 0:
    #         loss_loc = F.smooth_l1_loss(loc_p, loc_t_pos, reduction='sum')
    #     else:
    #         loss_loc = torch.tensor(0.0, device=self.device)
        

    #     # --------------------------------------
    #     # Calculate Confidence loss (CrossEntropy) with Hard Negative Mining
    #     # --------------------------------------
        
    #     batch_conf = conf_data.view(-1, num_classes) # flatten label dự đoán thành mảng 2 chiều (8723, 21)
    #     conf_t_flat = conf_t.view(-1) # truth label (8732)

    #     # Tính loss cross entropy cho toàn bộ bbox
    #     loss_conf_all = F.cross_entropy(batch_conf, conf_t_flat, reduction='none')
    #     loss_conf_all = loss_conf_all.view(batch_size, num_priors)
    #     # loss-conf_all trả về mỗi box có error bao nhiêu

    #     # Tính cho positive box
    #     num_pos = pos_mask.long().sum(dim=1, keepdim=True) # số positive box của mỗi ảnh

    #     _, loss_idx = loss_conf_all.sort(1, descending=True)
    #     _, idx_rank = loss_idx.sort(1)

    #     num_neg = torch.clamp(self.neg_pos * num_pos, max=num_priors) # số neg box = pos*3 là min, max = num_pos chỗ này đáng lẽ phải để max=pos*3
        
    #     print(idx_rank.shape)
    #     print(num_neg.shape)
    #     neg_mask = idx_rank < num_neg.expand_as(idx_rank) # lấy được những box neg với loss lớn nhất, 

    #     mask = (pos_mask + neg_mask).gt(0).view(-1) # chuyển thành mảng 1 chiều [True, False,.....]
    #     conf_t_pre = conf_data.view(-1, num_classes)[mask]
    #     conf_t_label_ = conf_t.view(-1)[mask]

    #     loss_conf = F.cross_entropy(conf_t_pre, conf_t_label_, reduction="sum")

    #     # total loss = loss_loc + loss_conf
    #     N = num_pos.sum()
    #     loss_loc = loss_loc/N
    #     loss_conf = loss_conf/N

    #     return loss_loc, loss_conf

    def forward(self, predictions, targets):
        loc_data, conf_data, dbox_list = predictions
        batch_size = loc_data.size(0)
        num_priors = loc_data.size(1)
        num_classes = conf_data.size(2)

        conf_t = torch.zeros((batch_size, num_priors), dtype=torch.long, device=self.device)
        loc_t = torch.zeros((batch_size, num_priors, 4), dtype=torch.float, device=self.device)

        dbox = dbox_list.to(self.device)
        variances = [0.1, 0.2]

        for idx in range(batch_size):
            truths = targets[idx][:, :-1]
            labels = targets[idx][:, -1]
            match(
                threshold=self.jaccard_threshold,
                truths=truths,
                priors=dbox,
                variances=variances,
                labels=labels,
                loc_t=loc_t,
                conf_t=conf_t,
                idx=idx
            )

        pos_mask = conf_t > 0

        # === Localization loss (Smooth L1) ===
        pos_idx = pos_mask.unsqueeze(2).expand_as(loc_t)
        if pos_mask.sum() > 0:
            loc_p = loc_data[pos_idx].contiguous().view(-1, 4)
            loc_t_pos = loc_t[pos_idx].contiguous().view(-1, 4)
            loss_loc = F.smooth_l1_loss(loc_p, loc_t_pos, reduction='sum')
        else:
            loss_loc = torch.tensor(0.0, device=self.device)
        # === Confidence loss (Cross Entropy + Hard Negative Mining) ===
        batch_conf = conf_data.view(-1, num_classes)
        conf_t_flat = conf_t.view(-1)

        loss_conf_all = F.cross_entropy(batch_conf, conf_t_flat, reduction='none').view(batch_size, num_priors)
        num_pos = pos_mask.long().sum(dim=1, keepdim=True)

        # Sort and rank losses
        _, loss_idx = loss_conf_all.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        # Hard negative mining: select top 3x negatives
        num_neg = torch.clamp(self.neg_pos * num_pos, max=num_priors)
        neg_mask = idx_rank < num_neg.expand_as(idx_rank)

        # Combine positive + hard negative samples
        mask = (pos_mask + neg_mask).gt(0)
        conf_p = conf_data[mask.unsqueeze(2).expand_as(conf_data)].view(-1, num_classes)
        targets_weighted = conf_t[mask]

        loss_conf = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # === Normalize losses safely ===
        N = num_pos.sum().clamp(min=1)
        loss_loc /= N
        loss_conf /= N

        return loss_loc, loss_conf


if __name__ == "__main__":

    batch_size = 2          # 2 ảnh trong batch
    num_priors = 5          # 5 default boxes
    num_classes = 4         # 3 lớp (background + 2 object classes)

    # loc_data: (batch, num_priors, 4)
    loc_data = torch.randn(batch_size, num_priors, 4, requires_grad=True)

    # conf_data: (batch, num_priors, num_classes)
    conf_data = torch.randn(batch_size, num_priors, num_classes, requires_grad=True)

    # dbox_list: (num_priors, 4)
    # Default boxes giả định nằm trong [0,1] normalized coordinates
    dbox_list = torch.tensor([
        [0.5, 0.5, 0.2, 0.2],
        [0.3, 0.3, 0.4, 0.4],
        [0.7, 0.7, 0.3, 0.3],
        [0.2, 0.8, 0.2, 0.2],
        [0.8, 0.2, 0.2, 0.2],
    ])

    # targets: list of tensors, mỗi ảnh có [xmin, ymin, xmax, ymax, label]
    targets = [
        torch.tensor([
            [0.4, 0.4, 0.6, 0.6, 1],  # object class 1
            [0.75, 0.15, 0.95, 0.35, 2], # object class 2
        ]),
        torch.tensor([
            [0.2, 0.7, 0.4, 0.9, 1],  # object class 1
        ])
    ]

    # ======= Tạo loss và chạy forward =======
    criterion = MultiBoxLoss(device='cpu')

    loss_loc, loss_conf = criterion((loc_data, conf_data, dbox_list), targets)

    print("Localization Loss:", loss_loc.item())
    print("Confidence Loss:", loss_conf.item())
    print("Total Loss:", (loss_loc + loss_conf).item())