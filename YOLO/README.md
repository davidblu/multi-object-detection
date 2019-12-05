#  Multi-Object Detection using YOLO
Here we try two different backbone models for YOLO, *resnet50* and *vgg16_bn*. Furthermore, we analyze the performance of two different backbone models.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

To run the notebook, you need the following packages installed:

* PyTorch
* Matplotlib

### Installing

* Download the training, validation, and testing dataset [here](https://drive.google.com/drive/folders/1jLSm7vNvcXMfPVHEtjNCFNrsXIIy9_s5?usp=sharing) on Google Drive.

* Download the trained model [here](https://drive.google.com/drive/folders/1oUUUSJq1h5hZItsM-euEJjwOLvCKaQck?usp=sharing) on Google Drive.

## Code organization

    .
    ├── demo.ipynb              # To showing how well our model perform on a single image
    ├── train.ipynb             # For a full traininng on the VOC 2007 & 2012 training dataset
    ├── result                  # Folder contains predicted result / training loss / validation loss 
    ├── src                     # Python scripts
    │   ├── dataset.py          # Define user dataset
    │   ├── eval_voc.py         # Evaluate the performance of the model
    │   ├── net.py              # Define various VGG models
    │   ├── nntools.py          # Useful modules for deep learning
    │   ├── plot.py             # Plot training / validation loss
    │   ├── predict.py          # Make the prediction on single image
    │   ├── resnet_yolo.py      # Define various resnet models
    │   ├── xml_2_txt.py        # Convert .xml to .txt file
    │   └── yoloLoss.py         # Module implementing yolo loss function 
    └── README.md



## Running the tests

### Executable Files

For a quick demo, run the Jupyter Notebook `demo.ipynb`.
For a full training on the dataset, run the Jupyter Notebook `train.ipynb`.


## Authors

* **Arthur Hsieh** - *Initial work* - [arthur960304](https://github.com/arthur960304)
* **Louis Lu** - *Initial work* - [louis910](https://github.com/louis910)

## References
[1] - [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)
