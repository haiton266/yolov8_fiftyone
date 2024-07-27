# YOLOv8 to FiftyOne Visualization

This repository is for converting a YOLOv8 dataset format to FiftyOne, enabling dataset visualization and problem identification.

## Installation

Install the required packages by running:

```bash
pip install -r requirements.txt
```

### File structure

- `main.py`: use for detection task without evaluation
- `detection_fiftyone.ipynb`: use for detection task
- `segmentation_fiftyone.ipynb`: use for segmentation task

### Usage

- Run result of model by:

```
    yolo task=detect mode=predict model=[model].pt source=[images] save_txt=True save_conf=True

```

- Edit some path:

  `dataset_dir`: Link dataset with structure folder:

  ```
  dataset_dir/
  ├── images
  ├── labels
  └── prediction
  ```

- `name_prediction`: Name of the prediction (Rename to easily distinguish between models)
- `classes`: List of model classes

### Reference

For more information, visit the [FiftyOne YOLOv8 Tutorial](https://docs.voxel51.com/tutorials/yolov8.html).
