import torch
from make_data_path import make_datapath
from dataset import MyDataSet, my_collate_fn
from transform import DataTransform
from extract_inform import AnnotationExtractor
import torch.utils.data as data
from model import SSD
import torch.nn as nn
from multibox_loss import MultiBoxLoss
from torch.optim import SGD
import time
import pandas as pd

torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root_path = './datasets/VOC2012/'
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath(root_path=root_path)

classes = ["aeroplane", "bicycle", "bird",  "boat", "bottle", 
               "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]

color_mean = (104, 117, 123) # color mean chuẩn hóa data giúp mạng hội tụ nhanh hơn tránh bùng nổ hay biến mất gradient
input_size = 300 # cần đầu vào SSD là 300 pixel

# Tạo dataset và data loader
data_transform = DataTransform(input_size=input_size, color_mean=color_mean)
train_dataset = MyDataSet(train_img_list, train_anno_list, phase='train', transform=data_transform, anno_xml=AnnotationExtractor(classes))
val_dataset = MyDataSet(val_img_list, val_anno_list, phase='val', transform=data_transform, anno_xml=AnnotationExtractor(classes))

batch_size = 32
train_dataloader = data.DataLoader(
                                dataset=train_dataset, 
                                batch_size=batch_size, 
                                shuffle=True, 
                                collate_fn=my_collate_fn
                                )

val_dataloader = data.DataLoader(
                                dataset=val_dataset, 
                                batch_size=batch_size, 
                                shuffle=False , 
                                collate_fn=my_collate_fn
                                )


dataloader_dict = {
    "train": train_dataloader,
    "val": val_dataloader
}


# CREATE NETWORK
cfgs = {
    "num_classes": 21,
    "input_size": 300,
    "bbox_aspect_num": [4, 6, 6, 6, 4, 4],
    "feature_maps": [38, 19, 10, 5, 3, 1],
    "steps": [8, 16, 32, 64, 100, 300],

    "min_sizes": [30, 60, 111, 162, 213, 264], # min_sizes for default boxes
    "max_sizes": [60, 111, 162, 213, 264, 315], # max_sizes for default boxes
    "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]], # aspect ratios for default boxes
}

net = SSD(phase='train', cfg=cfgs).to(device)

vgg_weight = torch.load("./datasets/weights/vgg16_reducedfc.pth", map_location="cpu")

net.vgg.load_state_dict(vgg_weight)

# load thông số khởi tạo
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)  # thuật toán sinh ra các số để khởi tạo layer
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# Kaiming He init
net.extras.apply(weights_init)
net.locate.apply(weights_init)
net.confidence.apply(weights_init)

# CREATE LOSS FUCNTION
criterion = MultiBoxLoss(jaccard_threshold=0.5, neg_pos=3, device=device)

# OPTIMIZER
optimizer = SGD(net.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4 ) # ngoài ra có thể đưa chỉ những parameter ở lớp nào lớp nào vào thôi, cái này advance


# training and validation

# def train_model(net, dataloader_dict, criterion, epochs):

#     # move net work to device
#     net.to(device)

#     iteration = 1
#     epoch_train_loss = 0.0
#     epoch_val_loss = 0.0

#     logs = []

#     for epoch in range(epochs+1):
#         time_epoch_start = time.time()
#         time_iter_start = time.time()

#         print("---"*20)
#         print("Epoch {}/{}".format(epoch+1, epochs))
#         print("---"*20)

#         for phase in ["train", "test"]:
#             if phase == 'train':
#                 net.train()  # Gọi main trong class mẹ nn.Module, enable thông số để lưu các thông số về đạo hàm
#                 print("Training")
#             else:
#                 if (epoch+1) % 10 == 0:
#                     net.eval()
#                     print("---"*10)
#                     print("Validation")
#                 else:
#                     continue
        
#             for images, targets in dataloader_dict[phase]:
#                 # move to gpu or cpu
#                 images = images.to(device)
#                 targets = [ann.to(device) for ann in targets]

#                 # init optimizer
#                 optimizer.zero_grad()

#                 # forward
#                 with torch.set_grad_enabled(phase=='train'):
#                     outputs = net(images)

#                     loss_locate, loss_confidence = criterion(outputs, targets)
#                     loss = loss_confidence + loss_locate

#                     if phase == 'train':
#                         loss.backward()
#                         torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
#                         optimizer.step() # update parameters

#                         if iteration % 2 == 0:
#                             time_iter_end = time.time()
#                             duration = time_iter_end - time_iter_start
#                             print("Iteration {} || Loss {:.4f} || 2iter: {:.4f}".format(iteration, loss.item(), duration))

#                             time_iter_start = time.time()

#                         epoch_train_loss += loss.item()
#                         iteration += 1
                    
#                     else:
#                         epoch_val_loss += loss.item()
        
#         time_epoch_end = time.time()
#         print("---"*20)
#         print("Epoch {} || Epoch_train_loss {:.4f} || Epoch_val_loss: {:.4f}".format(epoch+1, epoch_train_loss, epoch_val_loss))
#         print("Duration: {:.4f} sec".format(time_epoch_end-time_epoch_start))
#         time_epoch_start = time.time()

#         log_epoch = {
#             "epoch": epoch + 1,
#             "train_loss": epoch_train_loss,
#             "val_loss": epoch_val_loss
#         }
#         logs.append(log_epoch)

#         df = pd.DataFrame(logs)
#         df.to_csv("./data/ssd_loss.csv")

#         epoch_train_loss = 0.0
#         epoch_val_loss = 0.0

#         # save model pth

#         if (epoch+1) % 10 == 0:
#             torch.save(net.state_dict(), "./data/weights/ssd300_"+ str(epoch+1) + ".pth")


import math
import torch

def train_model(net, dataloader_dict, criterion, epochs):
    # move net to device
    net.to(device)

    # It's safer to create optimizer after moving model to device (optional)
    # But if you want to keep your original optimizer, you can leave it.
    # Here we re-create optimizer to be explicit:
    optimizer = SGD(net.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)

    # optional: LR warmup scheduler (simple)
    def adjust_lr(optim, iters_done, base_lr=1e-4, warmup_iters=1000):
        if iters_done < warmup_iters:
            lr = base_lr * float(iters_done + 1) / float(warmup_iters)
            for g in optim.param_groups:
                g['lr'] = lr

    torch.backends.cudnn.benchmark = True

    iteration = 1
    logs = []
    torch.autograd.set_detect_anomaly(False)  # set True only when debugging

    for epoch in range(epochs):
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        time_epoch_start = time.time()

        print("==="*10)
        print("Epoch {}/{}".format(epoch+1, epochs))
        print("==="*10)

        for phase in ["train", "val"]:   # you used 'test' before; your dataloader key is 'val'
            if phase == 'train':
                net.train()
                print("Training")
            else:
                net.eval()
                print("Validation")

            dataloader = dataloader_dict["train"] if phase == "train" else dataloader_dict["val"]

            for images, targets in dataloader:
                # move to device
                images = images.to(device)
                targets = [ann.to(device) for ann in targets]

                optimizer.zero_grad()

                # optional LR warmup
                adjust_lr(optimizer, iteration, base_lr=1e-4, warmup_iters=500)

                with torch.set_grad_enabled(phase=='train'):
                    outputs = net(images)  # (loc, conf, dbox_list)

                    # quick NaN check on outputs
                    loc_out, conf_out, dbox_list = outputs
                    if torch.isnan(loc_out).any() or torch.isnan(conf_out).any():
                        print(f"⚠️ NaN in network outputs at iter {iteration}. Skipping batch.")
                        print("loc_out NaN:", torch.isnan(loc_out).any().item(), 
                              "conf_out NaN:", torch.isnan(conf_out).any().item())
                        # optionally print stats
                        print("loc_out stats:", torch.min(loc_out).item(), torch.max(loc_out).item())
                        print("conf_out stats:", torch.min(conf_out).item(), torch.max(conf_out).item())
                        continue

                    loss_locate, loss_confidence = criterion(outputs, targets)

                    # check loss components
                    if torch.isnan(loss_locate) or torch.isnan(loss_confidence):
                        print(f"⚠️ NaN loss at iter {iteration}. Skipping batch and dumping diagnostics.")
                        print("loss_locate:", loss_locate, "loss_confidence:", loss_confidence)
                        # diagnose conf/loc/targets
                        with torch.no_grad():
                            # inspect conf and loc predictions and target ranges
                            print("conf_out mean/std:", conf_out.mean().item(), conf_out.std().item())
                            print("loc_out mean/std:", loc_out.mean().item(), loc_out.std().item())
                            # check targets validity
                            for i, t in enumerate(targets):
                                if t.numel() == 0:
                                    print(f" target[{i}] empty")
                                else:
                                    boxes = t[:, :4]
                                    mins = boxes.min()
                                    maxs = boxes.max()
                                    print(f" target[{i}] x range: {mins.item():.3f}..{maxs.item():.3f}")
                        continue

                    loss = loss_confidence + loss_locate

                    if phase == 'train':
                        # backward with try/except (to catch anomalies)
                        try:
                            loss.backward()
                        except RuntimeError as e:
                            print(f"Exception during backward at iter {iteration}: {e}")
                            # when anomaly, you might enable detect_anomaly for one batch
                            torch.autograd.set_detect_anomaly(True)
                            try:
                                loss.backward()
                            except Exception as e2:
                                print("Still failing with detect_anomaly, skipping batch. Error:", e2)
                                torch.autograd.set_detect_anomaly(False)
                                continue
                            torch.autograd.set_detect_anomaly(False)

                        # clip grads
                        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10)

                        optimizer.step()

                        # logging
                        if iteration % 50 == 0:
                            print(f"Iteration {iteration} || Loss {loss.item():.4f}")

                        epoch_train_loss += loss.item()
                        iteration += 1
                    else:
                        epoch_val_loss += loss.item()

        # end epoch
        time_epoch_end = time.time()
        print("Epoch {} || train_loss {:.4f} || val_loss {:.4f}".format(
            epoch+1, epoch_train_loss, epoch_val_loss))
        print("Duration: {:.2f}s".format(time_epoch_end - time_epoch_start))

        logs.append({
            "epoch": epoch+1,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss
        })
        pd.DataFrame(logs).to_csv("./datasets/ssd_loss.csv", index=False)

        # save checkpoint periodically
        if (epoch+1) % 10 == 0:
            torch.save(net.state_dict(), f"./datasets/weights/ssd300_epoch{epoch+1}.pth")

    print("Training finished.")
    
num_epoch = 32
train_model(net=net, dataloader_dict=dataloader_dict, criterion=criterion, epochs=num_epoch)


