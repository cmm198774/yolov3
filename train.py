import torch
import numpy as np
import pandas as pd
import time
from my_utils import *
from terminaltables import AsciiTable
import datetime
import sys
from torch.utils.data import DataLoader
from my_datasets import ListDataset
from yolo_v3 import Darknet
import warnings

cfg_root_path='.//config//'
raw_path='.//raw_data//coco2014//'
weight_path='.//weight//'
log_path='.//log//'
warnings.filterwarnings("ignore")

def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()
    img_path,label_path=path
    # Get dataloader
    dataset = ListDataset(img_folder=img_path,label_path=label_path,img_size=img_size, augment=False, multiscale=False)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (paths,imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        imgs=imgs.to(next(model.parameters()).device)
        targets=targets
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs).to('cpu')
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class

#训练部分主要参数
def train():
    #主要参数定义
    epochs=1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size=416
    batch_size=20
    multiscale_training=False
    augment=True
    config_path=cfg_root_path+'yolov3.cfg'
    log_file_path=log_path+'yolo3.log'
    train_img_path=raw_path+'images//train2014//'
    train_label_path=raw_path+'annotations//train_info_2014.csv'
    class_label_path=raw_path+'annotations//coco2014.names'
    checkpoint_interval=50
    #gradient_accumulations=1
    evaluation_interval=1001
    eval_img_path=raw_path+'images//val2014//'
    eval_label_path=raw_path+'annotations//val_info_2014.csv'
    check_point_path='.//checkpoint//'
    pretrained_weights_path=None

    class_names=load_classes(class_label_path)
    model = Darknet(config_path=config_path).to(device)
    model.apply(weights_init_normal)
    logger = Logger(log_file_path)
    # If specified we start from checkpoint
        # If specified we start from checkpoint
    if pretrained_weights_path:
        if pretrained_weights_path.endswith(".pth"):
            model.load_state_dict(torch.load(pretrained_weights_path))
        else:
            model.load_darknet_weights(pretrained_weights_path)

    # Get dataloader
    dataset = ListDataset(img_folder=train_img_path,label_path=train_label_path,img_size=img_size, augment=augment, multiscale=multiscale_training)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )
    print('dataset finished,dataloader size=%d'%len(dataloader))
    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        for batch_i, (img_path,imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i
            print('curr epoch=%d,batch=%d'%(epoch+1,batch_i))
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            loss, outputs = model(imgs, targets)
            loss.backward()

        #if batches_done % gradient_accumulations:
            # Accumulates gradient before each step
            optimizer.step()
            optimizer.zero_grad()
            
            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)
            

        if (epoch+1) % evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=(eval_img_path,eval_label_path),
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=img_size,
                batch_size=batch_size,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")
            
        if epoch % checkpoint_interval == 0:
            torch.save(model.state_dict(), check_point_path+"yolov3_ckpt_%d.pth" % epoch)
    torch.cuda.empty_cache()

if __name__=='__main__':
    train()