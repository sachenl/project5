# **Project 5 final**
Student name: Zhiqiang Sun

Student pace: self paced

## **Overview**
The skin cancer dataset contains many medical images that show various kinds of skin cancer. In this project, we will analyze and visualize the relationship between cancer and age and the location of the body. Furthermore, we will use machine learning to train a model that can distinguish the cancer type by given images. 

## **Dataset**
The whole dataset were download from kaggle (https://www.kaggle.com/code/rakshitacharya/skin-cancer-data/data). The folder contains several csv files and two images folder. All the name of images were named with image id which can be found in the metadata excel file. There are several other hinist csv file which include the pixels information of corresponding images in different resolusion. In this project, we will focus on the information from the metadata. Also, when we creat the model, we will use the original images for higher resolusion, thus we will dismiss all the hmnist data this time. 

The data has seven different classes of skin cancer which are listed below :
1. Melanocytic nevi
2. Melanoma
3. Benign keratosis-like lesions
4. Basal cell carcinoma
5. Actinic keratoses
6. Vascular lesions
7. Dermatofibroma

In this project, I will try to train a model of 7 different skin cancer classes using Convolution Neural Network with Keras TensorFlow and then use it to predict the types of skin cancer with random images.
Here is the plan of the project step by step:



1. Import all the necessary libraries for this project
2. Make a dictionary of images and labels
3. Reading and processing the metadata
4. Process data cleaning
5. Exploring the data analysis
6. Train Test Split based on the data frame 
7. Creat and transfer the images to the corresponding folders 
8. Do image augmentation and generate extra images to the imbalanced skin types
9. Do data generator for training, validation, and test folders
10. Build the CNN model
11. Fitting the model
12. Model Evaluation
13. Visualize some random images with prediction



## 3. Reading and processing the metadata**
In this step, we have read the csv which had the information for all the patients and images. Afterthat, we made three more columns including the cancer type in full name, the label in skin cancers in digital and the path of image_id in the folder.


![fig1_meta](https://raw.githubusercontent.com/sachenl/project5/main/images/fig1_meta.png)

## **4. Process data cleaning**
In this part, we check the missing values for each column and fill them. We found there are 57 nulls in age column, we then filled them with the mean value of age.


## **5. Exploring the data analysis**
In this part, we briefly explored different features of the dataset, their distributions and counts.

As there is some duplecate lesion_id which belong to same patient, all the features except the image_id for them are same with each other.  Thus, we first find and remove the duplex in patient id  and then plot the distributions of each features.

#### plot distribution of features 'dx', 'dx_type',  'sex', 'localization'.



![](https://raw.githubusercontent.com/sachenl/project5/main/images/fig5.png)

We checked the distribution of columns 'dx', 'dx_type',  'sex', 'localization' for different patients. The graphs show that:
1. In dx features, the 'nv': 'Melanocytic nevi' case take more than 70% of the total cases. The number suggests that this dataset is an unbalanced dataset.
2. In dx_type features, the histogram suggests most of the cancer were confirmed in Follow-up and histo Histopathologic diagnoses.
3. The sex feature shows that the amount of male who had skin cancer is slight larger than female but still similar to each other.
4. The localization analysis shows that  lower extremity, back ,trunk abdomen and upper extremity are heavily compromised regions of skin cancer



### Creat dashboard to visualize the distribution of age for different types of skin cancer
![](https://raw.githubusercontent.com/sachenl/project5/main/images/ezgif.com-gif-maker.gif)


In general, most cancers happen between 35 to 70.  Age 45 is a high peak for patients to get a skin cancer.  Some types of skin cancer (vasc, nv) happen to those below 20, and others occur most after 30.


## 6. Train Test Split based on the data frame
We split the dataset to training (70%), validation (10%) and testing (20%) by train_test_split.

The shape of each dataset are showed below:
((7210, 9), (802, 9), (2003, 9))


## 7. Creat and transfer the images to the corresponding folders
We created the subfolders containing the train, Val, and test folder. In addition, we created a folder for all types of skin cancers in each of the folders. Finally, We transferred the images to the corresponding folder based on the data frame and the path in each image ID.


![](https://raw.githubusercontent.com/sachenl/project5/main/images/fig6.png)

##  8. Do image augmentation and generate extra images to the imbalanced skin types

The amounts of files in each training folder type tell us the images of nv are much higher than others. The imbalance of the training dataset might cause a high bias in model fitting. Thus we will generate some more images for other kinds of cancers. Here we use image augmentation to oversample the samples in all classes except nv. Here is a simple chart about the oversampling.

![](https://raw.githubusercontent.com/sachenl/project5/main/images/oversampling.png)



We firsted check the number of images in each categories. And generated more images with data augmentation.

![](https://raw.githubusercontent.com/sachenl/project5/main/images/fig7.png)



After data augmentation, we check again the numbers of images in each folder.

![](https://raw.githubusercontent.com/sachenl/project5/main/images/fig8.png)


The numers of files in each folders are in same levels.


## **9. Do data generator for training, validation, and test folders**

Generat the dator for all three datasets.
Found 31825 images belonging to 7 classes.

Found 802 images belonging to 7 classes.

Found 2003 images belonging to 7 classes.


##  10. Build the CNN model

WE build a CNN model base on the pretrained model 'xception'.



![](https://raw.githubusercontent.com/sachenl/project5/main/images/fig9.png)

## **11. Fitting the model**
We fit the training data to the model we created earlier



![](https://raw.githubusercontent.com/sachenl/project5/main/images/fig10.png)


![](https://raw.githubusercontent.com/sachenl/project5/main/images/fig11.png)


save('results_on_xception_final_2.h5')

## **12. Model Evaluation**
In this step we will check the testing accuracy and validation accuracy of our model,plot confusion matrix and also check the missclassified images count of each type.



![](https://raw.githubusercontent.com/sachenl/project5/main/images/fig12.png)


The accuracy of the model is 74.84% which is not bad at all.

#### Plot the confusion matrix
![](https://raw.githubusercontent.com/sachenl/project5/main/images/fig14.png)

#### plot the evaluation scores for each category
![](https://raw.githubusercontent.com/sachenl/project5/main/images/fig15.png)

The f1 score for nv class is highest and over 0.88. The f1-score on df, akiec and mel are less than 0.5 which sugessted that the prediction on these three type are less accurate.


![](https://raw.githubusercontent.com/sachenl/project5/main/images/fig16.png)

It seems that the maximum number of incorrect predicitons are features mel and then df and akiec. The nv has least misclassified type. 


#### Finally we plot several of images randomly selected from test folder and mark them with predict and actual case on the top. 
![](https://raw.githubusercontent.com/sachenl/project5/main/images/fig17.png)

## **Conclusion**

We can extract the information about skin cancer from the metadata and explore the distribution of various features. For example, the most often age of skin cancer occur is around 45.
We make one CNN model which can fit and predict the type of skin cancer well based on the images. The accuracy is 74.9% which is more efficient than detection with human eyes.
