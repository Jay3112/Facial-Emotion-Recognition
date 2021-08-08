# Facial-Emotion-Recognition (v1.0)
Basically, in this project I have developed a deep learning model wich is able to recognise basic emotions (such as  happiness, sadness, anger, surprise, fear and disgust) from human face image.

## Description
To implement facial emotion recogntion I have used sobel filtering technique for extracting edge features. Then I have used three deep convolutional neural networks (ResNet50, VGG16 and DenseNet121) for further feature extraction and classification. After getting confidence from three DCNN, avrage voting (ensemble learning technique) used and got final results. Using this model I got 66.35% accuracy for sobel filtered images, and if we pass raw data into model it is able to acheive 69.59% accuracy.


## Flow of the experiment
![Flow diagram](/images/logo.png)


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

### Menifest
```
RM_Proj_Data_Process.ipynb / RM_Proj_Data_Process.py :

  * Code for .zip file extraction to extract image data from zipfile. 
  * Code for retriving train and test image folders, code for resizing images from 48x48 pixels to 224x224 pixels.
  * Code for applying soble filter to resized image. 
  * Code for converting raw images and filtered images into arrays and store in .npy formate.

RM_Proj_NN.ipynb / RM_Proj_NN.py :

  * Code for loading .npy files from drive to load data and labels from drive.
  * Code for training ResNet50, VGG16 and DesneNet121 models with sobel filtered data and with raw data also.
  * Code for ensemble learning for sobel filtered data and raw data.
  
Images : This folder includes images of project flow, experiment results and example of data.

train_data.npy , train_labels.npy : Contains training image arrays and training label arrays.
test_data.npy  , test_labels.npy  : Contains testing  image arrays and training label arrays.

sobel_train.npy , sobel_test.npy  : Contains soebl filtered training and testing image arrays.

rawdata_vgg16.hdf5    : VGG16 model trained on rawdata (model architecture and weights).
rawdata_resnet50.hdf5 : ResNet50 model trained on rawdata (model architecture and weights).
rawdata_dense121.hdf5 : DenseNet121 model trained on rawdata (model architecture and weights).

sobel_filt_vgg16.hdf5    : VGG16 model trained on sobel filtered data (model architecture and weights).
sobel_filt_resnet50.hdf5 : ResNet50 model trained on sobel filtered data (model architecture and weights).
sobel_filt_dense121.hdf5 : DenseNet121 model trained on sobel filtered data (model architecture and weights).
```

### Dataset
* I have used FER2013 dataset as it contains real life images taken from videos and news any other sources. It is bit challanging due to occluded images with different light illumination.
* To download the datatset : https://www.kaggle.com/msambare/fer2013

![Example of FER2013 data](/images/logo.png)


### Executing program

Method 1 :
* If running environment is google colab just make folder on drive named "FER2013" and put downloaded dataset into it.
* Then download and run RM_Proj_Data_Process.ipynb, RM_Proj_NN.ipynb files one after other which will do all the work for you.

Method 2 :
* If using local machine download Open CV and then run RM_Proj_Data_Process.py, RM_Proj_NN.py one by one.
* **make sure you chnage all the path given in code according to your local machine**

Method 3 :
* If you want use .npy files which contains already preprocessed data.
* Download all the .npy files and run RM_Proj_NN.ipynb or RM_Proj_NN.py 

Method 4 :
* If you need very fast execution you can use .npy files along with .hdf5 files.
* Download all the .npy and .hdf5 files.
* Run modeules given below from file RM_Proj_NN.ipynb or RM_Proj_NN.py
  * Ensemble Learning for Rawdata (ResNet50 + VGG16 + DenseNet121)
  * Classification report and Confusion Matrix for Raw data
  * Ensemble Learning for Sobel filtered data (ResNet50 + VGG16 + DenseNet121)
  * Classification report and Confusion matrix for Sobel filtered data


## Help
If you are using google colab, low RAM can be the issue. To overcome that, run only one DCNN at a time and to run second DCNN restart the execution kernel. When you complete one run, it will save best trained model weights to your google drive in .hdf5 fromat so, you dont need to execute that part again and again.

## Authors
Jay Patel<br/>
In case of queary send an email on (pateljay311297@gmail.com)

## Version History
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments
