# NN_Classifier_Project

This is an image classifier project from the [Udacity AI Programming with Python](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089).
The classifier is currently trained with a data for [102 flowers](https://github.com/Blade-Storm/UdacityImageClassifier/blob/master/cat_to_name.json).
Until know it has only two pre-model arch supported Densenet121 and VGG19.

## Setup
We assume that you have familiarity with a Python package manager      

Python version: 3.7

Packages that you need to have:
   * argparse
   * torch
   * matplotlib
   * torchvision
   * PIL
   * numpy
   * json
   * time
   
## How to use
It's streat forward to use it. You will need guide the following steps:
###1. Train
You will need to train first ant then use your trained model to make inferences. To do the training you will 
need to use the default settings as show below or config pass the parameters.

``` python train.py```
    
Possible arguments that is possible to pass:
   * --data-directory
     * Path to the image files. The folder should contain: "train", "test" and "valid" folders;
     * Default is ``./flowers``
   * --save-dir
     * The relative path to the directory you wish to save the trained model's checkpoint to.
     * Default is ``./checkpoints``
   * --arch
     * Arch with the pre-trained model. In this version is supported densenet121 and vgg19 
     * Default is ```densenet121```
   * --learning-rate
     * Learning rate of the model that you are going to train
     * Default is ``0.0003``
   * --dropout
     * Dropout rate used during the training of the model
     * Default is ``0.005``
   * --hidden-layers
     * weights of the hidden layers that are going to use. This version suport "N" layers
     * Default is ``256``
     * the "N" layers can be passed like this example: ``512 256 128``
   * --epoch
     * Number of epoch the train mode are going to use
     * Default is ``5``
   * --batch-size
     * The size of the image batches you want to use for training
     * Default is ``32``
     
Example using the parameters:
```
python train.py --data-directory ./flowers --save-dir ./checkpoints --arch densenet121 --learning-rate 0.0003 --dropout 0.005 --hidden-layers 512 128 --epoch 6 --batch-size 32
```
###2. Doing inference 
After your model is trained we can infer the flower name. You can use the test folder as an example to predict it. 
To do the prediction you will need to run like this:
```
python predict.py --data-directory ./flowers/valid/1/image_06756.jpg --checkpoint ./checkpoints/checkpoint-30-densenet121.pth
```

Possible arguments:
  * --data-directory
    * Path to the image file that you want to infer.
  * --checkpoint
    * The relative path to the models checkpoint pth file
  * --top_k
    * Number of classes and the % of infer
    * Default is ``5``
  * --category-names
    * The json file contaning the classes and names of categories
    * Default is ``./cat_to_name.json``