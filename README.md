# Kaggle-Statoil-C-CORE-Iceberg-Classifier-Challenge
It's the record of my experience in the Kaggle competition, <a herf=https://www.kaggle.com/c/statoil-iceberg-classifier-challenge>Statoil-C-CORE-Iceberg-Classifier-Challenge</a>. </br>
## 1.Define the base Network.
I chose not to use pretrained network, such as `VGG16` etc. Because VGG16 or other pretrained network was mostly trained on `IMAGENET`, which includes mostly everyday objects, while in this conpetition, we need to classify iceberg from the radar data. So, I think it's a diffenert task. So, I trained my `vgg16-like` network from scratch.</br>
Here is my network's architecture.</br>
![image](https://github.com/mjDelta/Kaggle-Statoil-C-CORE-Iceberg-Classifier-Challenge/blob/master/imgs/network.png)</br>
It is a vgg16-like network with the same conv layers( but different channels, because in this task, the input bands is 75px, other than 224px. Besides, iceberg maybe have less features than the thousands of everyday objects).
Usually, conv layers is used for bands feature extraction. After Flatten operation, we merge it with the Dense layer which has the parameter inc_angle as the input.</br>
#### Besides, we still use pretrained model for stacking. And thanks to Yu Hai's kernel:https://www.kaggle.com/yuhaichina/single-model-vgg16-mobilenet-lb-0-1568-with-tf</br>
## 2.About the Preprocessing
In the preprocessing, we use many methods for the preparation of stacking. And these methods include minmax, Fourier Transformation, ROF de-noising, Morphological Transformation etc. </br>

## 3.About Feature Extraction
We chosen to use our own pretrained vgg16-like network to do the feature extraction. To do the next classification, we chosen scikit-learn's SVM.</br>

## 4.About the Ensemble Model
For Ensemble Modeling, we chosen blending and stacking.
#### Blending
For blending, we used pretrained VGG16's test results( with different data preprocessing) as single model. Then do Linear Regression with these single models, and got LB score 0.151.
#### Stacking 
For stacking,we used blending results, new defined network's results and SVM's results to do the stacking. Our best model came from the minmax_bestbase stacking method, and got LB score 0.1208.
