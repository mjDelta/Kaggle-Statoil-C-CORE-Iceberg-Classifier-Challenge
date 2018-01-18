# Kaggle-Statoil-C-CORE-Iceberg-Classifier-Challenge
It's the record of my experience in the Kaggle competition, <a herf="https://www.kaggle.com/c/statoil-iceberg-classifier-challenge">Statoil-C-CORE-Iceberg-Classifier-Challenge</a>. </br>
## 1.Define the base Network.
I chose not to use pretrained network, such as `VGG16` etc. Because VGG16 or other pretrained network was mostly trained on `IMAGENET`, which includes mostly everyday objects, while in this conpetition, we need to classify iceberg from the radar data. So, I think it's a diffenert task. So, I trained my `vgg16-like` network from scratch.</br>
Here is my network's architecture.</br>
![image](https://github.com/mjDelta/Kaggle-Statoil-C-CORE-Iceberg-Classifier-Challenge/blob/master/imgs/network.png)</br>

