# YOLOv3 Object Detection (PyTorch)

A PyTorch implementation of the YOLOv3 object detection model, trained on the COCO 2014 dataset with 80 object categories.

## Project Structure

```
├── yolo_v3.py          # Core YOLOv3 network (Darknet, YOLOLayer, cfg parser)
├── train.py            # Training script with evaluation & checkpointing
├── interface.py        # Batch inference script with detection visualization
├── config/
│   ├── yolov3.cfg      # YOLOv3 network architecture definition
│   ├── yolov3-tiny.cfg # YOLOv3-Tiny architecture definition
│   ├── coco.data       # COCO dataset configuration
│   └── custom.data     # Custom dataset configuration template
├── my_datasets/
│   ├── datasets.py     # Custom Dataset (ListDataset) for COCO data loading
│   └── RunCocoLabelCsv.py  # COCO annotation to CSV converter
├── my_utils/
│   ├── utils.py        # NMS, IoU, coordinate transforms, AP calculation
│   └── logger.py       # TensorBoard logging utility
├── scripts/
│   ├── YoloV3.ipynb    # Main Jupyter notebook
│   └── test.ipynb      # Testing notebook
├── raw_data/           # Raw COCO 2014 images and annotations
├── weight/             # Pre-trained Darknet weights
├── checkpoint/         # Saved model checkpoints (.pth)
├── detect/             # Detection output images
└── log/                # Training logs
```
raw_data and checkpoint data is uploaded to baidu net disk: https://pan.baidu.com/s/1zjSTes1c0lKJAssmKnP1kQ,secret code  b4wn

## Architecture

This implementation follows the original YOLOv3 design:

- **Backbone**: Darknet-53 with residual blocks
- **Neck**: Feature Pyramid Network (FPN) for multi-scale detection
- **Head**: 3 YOLO detection layers at 13x13, 26x26, and 52x52 scales
- **Input size**: 416x416 (configurable)
- **Classes**: 80 COCO categories
- **Anchors**: 9 default anchors (3 per scale), parsed from `.cfg` file

The network configuration is dynamically parsed from `config/yolov3.cfg`, so layer definitions can be modified without changing Python code.

## Requirements

```
torch
numpy
pandas
matplotlib
Pillow
tqdm
terminaltables
torchvision
```

## Dataset Setup

Place the COCO 2014 dataset under `raw_data/coco2014/`:

```
raw_data/coco2014/
├── images/
│   ├── train2014/
│   └── val2014/
└── annotations/
    ├── train_info_2014.csv
    ├── val_info_2014.csv
    └── coco2014.names
```

Use `my_datasets/RunCocoLabelCsv.py` to convert COCO JSON annotations to CSV format.

## Usage

### Training

```bash
python train.py
```

Key parameters (defined in `train.py`):

| Parameter | Default | Description |
|---|---|---|
| `epochs` | 1000 | Total training epochs |
| `batch_size` | 20 | Batch size |
| `img_size` | 416 | Input image size |
| `config_path` | `./config/yolov3.cfg` | Network config file |
| `pretrained_weights_path` | `None` | Path to pre-trained weights (.pth or .weights) |
| `checkpoint_interval` | 50 | Save checkpoint every N epochs |
| `evaluation_interval` | 1001 | Run validation every N epochs |

To resume from a checkpoint, set `pretrained_weights_path` to the desired `.pth` file.

### Batch Inference

```bash
python interface.py
```

This loads the model checkpoint, runs detection on all images in `raw_data/coco2014/images/train2014/`, and saves annotated results to `detect/`.

Key parameters (defined in `interface.py`):

| Parameter | Default | Description |
|---|---|---|
| `conf_thres` | 0.95 | Confidence threshold |
| `nms_thres` | 0.7 | NMS IoU threshold |
| `pretrained_weights_path` | `./checkpoint/yolov3_ckpt_950.pth` | Model checkpoint to load |

### Loading Pre-trained Weights

The model supports both PyTorch `.pth` and original Darknet `.weights` files:

```python
from yolo_v3 import Darknet

model = Darknet(config_path='config/yolov3.cfg')

# PyTorch checkpoint
model.load_state_dict(torch.load('checkpoint/yolov3_ckpt_950.pth'))

# Or original Darknet weights
model.load_darknet_weights('weight/yolov3.weights')
```

## Custom Dataset

To train on your own dataset:

1. Prepare annotations in CSV format matching the `train_info_2014.csv` schema (`image_id`, `file_name`, `bbox`, `category_id_modif`).
2. Create a `.names` file with class labels (one per line).
3. Update paths in `train.py`:
   ```python
   train_img_path = 'path/to/images/'
   train_label_path = 'path/to/annotations.csv'
   class_label_path = 'path/to/classes.names'
   ```
4. Adjust `num_classes` and anchor masks in `config/yolov3.cfg` (search for `classes=` and `mask=` in each `[yolo]` section).

Or use `config/create_custom_model.sh` to auto-generate a custom config.

## Evaluation Metrics

During validation, the following metrics are computed and logged via TensorBoard:

- **mAP** (mean Average Precision)
- **Precision / Recall** at IoU=0.5
- **F1 Score**
- Per-class AP breakdown table

## License

This project is based on the YOLOv3 architecture described in [Redmon & Farhadi (2018)](https://pjreddie.com/darknet/yolo/).
