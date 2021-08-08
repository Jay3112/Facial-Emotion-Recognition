# Facial-Emotion-Recognition (v 1.0)
![GitHub followers](https://img.shields.io/github/followers/Jay3112?style=social)
![Languages](https://img.shields.io/github/languages/count/Jay3112/Facial-Emotion-Recognition)</br>
Basically, in this project, I have developed a deep learning model which can recognize basic emotions (such as happiness, sadness, anger, surprise, fear, and disgust) from the human face image.

## Description
In the current era, deep learning and computer vision techniques are making human life easier and more comfortable. The development of faster and accurate algorithms made face recognition, object tracking, and many more task faster and easier. Facial emotion recognition(FER) is one of them. Using FER we can detect the mood of any person and that may help us to develop applications such as smart computer interfaces, autonomous driving, health management, and many others.</br>

To implement facial emotion recognition I have used Sobel filtering technique for extracting edge features. Then I have used three deep convolutional neural networks (ResNet50, VGG16, and DenseNet121) for further feature extraction and classification. After getting confidence from three DCNN, average voting (ensemble learning technique) was used and got final results. Using this model I got 66.35% accuracy for Sobel filtered images, and if we pass raw data into the model it can achieve 69.59% accuracy.


## Flow of the experiment
![Flow diagram](/Images/RM_Project_Flowchart.png)


## Getting Started

### Dependencies
* Libraries
  * Open CV
  * Keras
  * Matplotlib
  * Pandas
  * OS
  * ZipFile
  * Numpy
  * Tensorflow
  * Sklearn
* Hardware specifications
  * RAM : 14 to 20 GB
  * GPU : 15 GB or above
  * Disk space : 1 GB
* Language used
  * Python : version above 3.0
  
* **If possible use google colab which provides all the dependecies mentined above**

### Dataset
* I have used the FER2013 dataset as it contains real-life images taken from videos and news any other sources. It is a bit challenging due to occluded images with different light illumination.
* To download the dataset: https://www.kaggle.com/msambare/fer2013

Some example images from the dataset are given below</br>

![Example of FER2013 data](/Images/FER-2013.jpg)

### Menifest
```
RM_Proj_Data_Process.ipynb / rm_proj_data_process.py :

  * Code for .zip file extraction to extract image data from zip file. 
  * Code for retrieving train and test image folders, code for resizing images from 48x48 pixels to 224x224 pixels.
  * Code for applying Sobel filter to resized image. 
  * Code for converting raw images and filtered images into arrays and store in .npy formate.

RM_Proj_NN.ipynb / rm_proj_nn.py :

  * Code for loading .npy files from drive to load data and labels from the drive.
  * Code for training ResNet50, VGG16, and DesneNet121 models with Sobel filtered data and with raw data also.
  * Code for ensemble learning for Sobel filtered data and raw data.
  
Images: This folder includes images of project flow, experiment results, and examples of data.
```

### Executing program

Downloaded dataset from given link.</br>
*If using local machine download libraries given in dependencies.*</br>
*If the running environment is google colab just make a folder on the drive named "FER2013" and put the downloaded dataset into it.*

Method 1 (For colab):
* Download and run RM_Proj_Data_Process.ipynb, RM_Proj_NN.ipynb files one after other which will do all the work for you.

Method 2 (For local machine):
* Run rm_proj_data_process.py, rm_proj_nn.py one by one to execute the whole project in the local machine.
* **make sure you change all the paths given in code according to your local machine**


### Results
Accuracy and loss curves, confusion matrix, classification report are availbale in [Result images](Images/) folder.</br>
For more information about results [Result images info](Images/image_info.md).

## Help
If you are using google colab, low RAM can be the issue. To overcome that, run only one DCNN at a time, and to run a second DCNN restart the execution kernel. When you complete one run, it will save the best-trained model weights to your google drive in .hdf5 format so, you don't need to execute that part again and again.

## Authors
Jay Patel<br/>
In case of query send an email to (pateljay311297@gmail.com)

## Version History
* 0.1
    * Initial Release

## License
This project is not licensed.

## Project status
The project is completed.</br>
Still, we can try data augmentation and different networks and features to optimize this model.
