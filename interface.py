import torch
import numpy as np
import pandas as pd
import time
from my_utils import *
import datetime
from torch.utils.data import DataLoader
from my_datasets import ListDataset
from yolo_v3 import Darknet
from PIL import Image
import random
import warnings
from tqdm import tqdm

cfg_root_path='.//config//'
raw_path='.//raw_data//coco2014//'
weight_path='.//weight//'
log_path='.//log//'
check_point_path='.//checkpoint//'
warnings.filterwarnings("ignore")


#训练部分主要参数
def interface():
    #主要参数定义
    conf_thres=0.95
    nms_thres=0.7
    img_size=416
    batch_size=20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path=cfg_root_path+'yolov3.cfg'
    check_point_path='.//checkpoint//'
    pretrained_weights_path=check_point_path+f"yolov3_ckpt_950.pth"
    class_label_path=raw_path+'annotations//coco2014.names'
    classes=load_classes(class_label_path)
    model = Darknet(config_path=config_path).to(device)
    output_path='.//detect//'
    img_folder_path=raw_path+'images//train2014//'
    label_path=raw_path+'annotations//train_info_2014.csv'

    if pretrained_weights_path:
        if pretrained_weights_path.endswith(".pth"):
            model.load_state_dict(torch.load(pretrained_weights_path))
        else:
            model.load_darknet_weights(pretrained_weights_path)

    model.eval()  # Set in evaluation mode

    dataset = ListDataset(img_folder=img_folder_path,label_path=label_path,img_size=img_size, augment=False, multiscale=False,is_train=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs,targets) in enumerate(tqdm(dataloader)):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, conf_thres, nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        #print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)
        

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    #print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        #print("(%d) Image: '%s'" % (img_i, path))
        detections=detections.to('cpu')
        # Create plot
        img = np.array(Image.open(path))
    #     img, pad = pad_to_square( transforms.ToTensor()(img), 0)
    #     img=img.permute(1,2,0).numpy()
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        
        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                #print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )

        # Save generated image with detections
        plt.axis("off")
    #     plt.gca().xaxis.set_major_locator(NullLocator())
    #     plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("//")[-1].split(".")[0]
        plt.savefig(output_path+f"{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()
    
if __name__=='__main__':
    interface()