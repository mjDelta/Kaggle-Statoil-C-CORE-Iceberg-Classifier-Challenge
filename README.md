# Kaggle-Statoil-C-CORE-Iceberg-Classifier-Challenge
It's the record of my experience in the Kaggle competition, <a herf="https://www.kaggle.com/c/statoil-iceberg-classifier-challenge">Statoil-C-CORE-Iceberg-Classifier-Challenge</a>. </br>
## 1.Define the base Network.
I chose not to use pretrained network, such as `VGG16` etc. Because VGG16 or other pretrained network was mostly trained on `IMAGENET`, which includes mostly everyday objects, while in this conpetition, we need to classify iceberg from the radar data. So, I think it's a diffenert task. So, I trained my `vgg16-like` network from scratch.</br>
Here is my network's architecture.</br>
![image](https://github.com/mjDelta/Kaggle-Statoil-C-CORE-Iceberg-Classifier-Challenge/blob/master/imgs/network.png)</br>
It is a vgg16-like network with the same conv layers( but different channels, because in this task, the input bands is 75*75, other than 224*224. Besides, iceberg maybe have less features than the thousands of everyday objects).
Usually, conv layers is used for bands feature extraction. After Flatten operation, we merge it with the Dense layer which has the parameter inc_angle as the input.</br>
#### Besides, we still use pretrained model for stacking.</br>
## 2.About the Preprocessing
In the preprocessing, we use many methods for the preparation of stacking. And these methods include minmax, Fourier Transformation, ROF de-noising, Morphological Transformation etc. </br>

## 3.About Feature Extraction
We chosen to use our own pretrained vgg16-like network to do the feature extraction. To do the next classification, we chosen scikit-learn's SVM.</br>


