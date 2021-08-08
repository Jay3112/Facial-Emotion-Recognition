# Facial-Emotion-Recognition (v1.0)
Basically, in this project I have developed a deep learning model wich is able to recognise basic emotions (such as  happiness, sadness, anger, surprise, fear and disgust) from human face image.


## Description
To implement facial emotion recogntion I have used sobel filtering technique for extracting edge features. Then I have used three deep convolutional neural networks (ResNet50, VGG16 and DenseNet121) for further feature extraction and classification. After getting confidence from three DCNN, avrage voting (ensemble learning technique) used and got final results. Using this model I got 66.35% accuracy for sobel filtered images, and if we pass raw data into model it is able to acheive 69.59% accuracy.


## Flow of the experiment
![Flow diagram](/images/logo.png)


## Getting Started

### Dependencies

* RAM : 14 to 20 GB
* Disk space : 1 GB
* Python : version above 3.0
* Open CV
* **If possible use google colab which provides all the dependecies mentined above**

### Menifest
* **RM_Proj_Data_Process.ipynb / RM_Proj_Data_Process.py :**
  * Code for .zip file extraction to extract image data from zipfile. 
  * Code for retriving train and test image folders, code for resizing images from 48x48 pixels to 224x224 pixels.
  * Code for applying soble filter to resized image. 
  * Code for converting raw images and filtered images into arrays and store in .npy formate.
* **RM_Proj_NN. ipynb / RM_Proj_NN.py :**
  * Code for loading .npy files from drive to load data and labels from drive.
  * Code for train ResNet50, VGG16 and DesneNet121 models

### Dataset

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
