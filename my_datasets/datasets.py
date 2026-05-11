import torch
import numpy as np
import pandas as pd
from my_utils import *
from torch.utils.data import Dataset
import json
import random
from torchvision import transforms
from PIL import Image
import os

#数据加载模块
def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = torch.nn.functional.pad(img, pad, "constant", value=pad_value)

    return img, pad


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets

def resize(image, size):
    image = torch.nn.functional.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

class ListDataset(Dataset):
    def __init__(self, img_folder,label_path=None, img_size=416, augment=True,multiscale=True,is_train=True):
        self.img_folder=img_folder
        self.label_info=pd.read_csv(label_path,header=0)
        self.image_info=self.label_info[['image_id','file_name']].drop_duplicates(subset=['image_id'])
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.multiscale=multiscale
        self.is_train=is_train

    def __getitem__(self, index):

        # ---------
        #  获取image数据
        # ---------
        image_id,file_name = self.image_info.iloc[index][['image_id','file_name']]
        img_path = os.path.join(self.img_folder,file_name.rstrip())
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        
        # ---------
        #  获取图片标注以及类别
        # ---------
        boxes=torch.tensor([json.loads(x) for x in self.label_info[self.label_info['image_id']==image_id]['bbox'].values])

        # 当img的维度不满3的时候
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))
        #plt.subplot(3,2,1),plt.imshow(img.permute(1,2,0).numpy()),plt.title('Original Image')
        _, h, w = img.shape
        
#         #将图片保存成需要的大小
#         if h<=w:
#             transform_size=transforms.Resize([int(h*self.img_size/w),self.img_size])
#             boxes[:,0:4]=boxes[:,0:4]*self.img_size/w

#         else:
#             transform_size=transforms.Resize([self.img_size,int(w*self.img_size/h)])
#             boxes[:,0:4]=boxes[:,0:4]*self.img_size/h
#         img=transform_size(img)
#         _, h, w = img.shape
        
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape
        targets = None
        if self.is_train:
            labels=torch.tensor(self.label_info[self.label_info['image_id']==image_id]['category_id_modif'].values)
            # Extract coordinates for unpadded + unscaled image
            x1 = boxes[:,0]
            y1 = boxes[:,1]
            x2 = boxes[:,0]+boxes[:,2]
            y2 = boxes[:,1]+boxes[:,3]
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # # Returns (x, y, w, h)
            boxes[:, 0] = x1
            boxes[:, 1] = y1
            #output输出box格式
            output_boxes=torch.zeros_like(boxes)
            output_boxes[:,0]=(boxes[:,0]+boxes[:,2]/2)/padded_w
            output_boxes[:,1]=(boxes[:,1]+boxes[:,3]/2)/padded_h
            output_boxes[:,2]=boxes[:,2]/padded_w
            output_boxes[:,3]=boxes[:,3]/padded_h
            targets = torch.zeros((len(boxes), 6))
            targets[:,1]=labels
            targets[:,2:]=output_boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)
     
        return img_path,img,targets
    
    def collate_fn(self, batch):
        img_paths,imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return img_paths,imgs, targets

    def __len__(self):
        return len(self.image_info)
