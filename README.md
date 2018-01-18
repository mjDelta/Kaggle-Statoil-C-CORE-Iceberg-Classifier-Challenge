# Kaggle-Statoil-C-CORE-Iceberg-Classifier-Challenge
It's the record of my experience in the Kaggle competition, Statoil-C-CORE-Iceberg-Classifier-Challenge. </br>
## 1.Define the base Network.
I chose not to use pretrained network, such as `VGG16` etc. Because VGG16 or other pretrained network was mostly trained on `IMAGENET`, which includes mostly everyday objects, while in this conpetition, we need to classify iceberg from the radar data. So, I think it's a diffenert task. So, I trained my `vgg16-like` network from scratch.</br>
Here is my network's architecture.

