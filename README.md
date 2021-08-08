# Facial-Emotion-Recognition (v1.0)
Basically, in this project, I have developed a deep learning model which can recognize basic emotions (such as happiness, sadness, anger, surprise, fear, and disgust) from the human face image.

## Description
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
RM_Proj_Data_Process.ipynb / RM_Proj_Data_Process.py :

  * Code for .zip file extraction to extract image data from zip file. 
  * Code for retrieving train and test image folders, code for resizing images from 48x48 pixels to 224x224 pixels.
  * Code for applying Sobel filter to resized image. 
  * Code for converting raw images and filtered images into arrays and store in .npy formate.

RM_Proj_NN.ipynb / RM_Proj_NN.py :

  * Code for loading .npy files from drive to load data and labels from the drive.
  * Code for training ResNet50, VGG16, and DesneNet121 models with Sobel filtered data and with raw data also.
  * Code for ensemble learning for Sobel filtered data and raw data.
  
Images: This folder includes images of project flow, experiment results, and examples of data.

train_data.npy, train_labels.npy: Contains training image arrays and training label arrays.
test_data.npy  , test_labels.npy: Contains testing image arrays and training label arrays.

sobel_train.npy, sobel_test.npy: Contains Sobel filtered training and testing image arrays.

rawdata_vgg16.hdf5: VGG16 model trained on raw data (model architecture and weights).
rawdata_resnet50.hdf5: ResNet50 model trained on raw data (model architecture and weights).
rawdata_dense121.hdf5 : DenseNet121 model trained on rawdata (model architecture and weights).

sobel_filt_vgg16.hdf5: VGG16 model trained on Sobel filtered data (model architecture and weights).
sobel_filt_resnet50.hdf5 : ResNet50 model trained on sobel filtered data (model architecture and weights).
sobel_filt_dense121.hdf5 : DenseNet121 model trained on sobel filtered data (model architecture and weights).
```

### Executing program

Method 1 :
* If the running environment is google colab just make a folder on the drive named "FER2013" and put the downloaded dataset into it.
* Then download and run RM_Proj_Data_Process.ipynb, RM_Proj_NN.ipynb files one after other which will do all the work for you.

Method 2 :
* If using local machine download Open CV and then run RM_Proj_Data_Process.py, RM_Proj_NN.py one by one.
* **make sure you change all the paths given in code according to your local machine**

Method 3 :
* If you want to use .npy files which contain already preprocessed data.
* Download all the .npy files and run RM_Proj_NN.ipynb or RM_Proj_NN.py 

Method 4 :
* If you need very fast execution you can use .npy files along with .hdf5 files.
* Download all the .npy and .hdf5 files.
* Run modeules given below from file RM_Proj_NN.ipynb or RM_Proj_NN.py
  * Ensemble Learning for Rawdata (ResNet50 + VGG16 + DenseNet121)
  * Classification report and Confusion Matrix for Raw data
  * Ensemble Learning for Sobel filtered data (ResNet50 + VGG16 + DenseNet121)
  * Classification report and Confusion matrix for Sobel filtered data

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

## Acknowledgments
