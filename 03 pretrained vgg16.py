# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 16:19:40 2018

@author: ZMJ
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from os.path import join as opj
import keras
import tensorflow as tf
from matplotlib import pyplot as plt
from numpy import *
import time
from scipy.ndimage import gaussian_filter
from skimage import img_as_float
from skimage.morphology import reconstruction
from preprocessing import *
def show_demo_pics(X_band_3,X_band_4,X_band_5,X_train,show=False):
    if show:
        plt.subplots(22)
        plt.subplot(221)
        plt.imshow(X_band_3[0])
        plt.subplot(222)
        plt.imshow(X_band_4[0])
        plt.subplot(223)
        plt.imshow(X_band_5[0])
        plt.subplot(224)
        plt.imshow(X_train[0])
        plt.show()
def turn_logloss_to_modelweights(logloss):
    logloss_=[loss-0.1599 for loss in logloss]
    sum_=np.sum(logloss_)
    weights=[loss/sum_ for loss in logloss_]
    return weights
    
##计算单个模型预测结果的皮尔逊相关系数
def compute_correlation(file_paths):    
    df=pd.DataFrame()
	
    for i,file_path in enumerate(file_paths):
        df_temp=pd.read_csv(opj("submissions",file_path),header=0)
        if i==0:
            df["id"]=df_temp["id"]
        df[file_path[10:-4]]=df_temp["is_iceberg"]
#    print("模型之间的相关系数表如下：")
#    print(df.corr())
    df.corr().to_csv("corr.csv")
   
##计算皮尔逊相关系数
file_paths=[
#            "sub_figure01_100.csv","sub_figure01_133.csv","sub_figure02_58.csv","sub_figure02_101.csv","sub_figure02_123.csv",\
#            "sub_figure02_170.csv","sub_figure04_79.csv","sub_figure04_122.csv",\
#            "sub_figure04_191.csv","sub_figure04_213.csv","sub_figure04_253.csv","sub_figure04_307.csv",\
#            "sub_figure04_350.csv",\
#            "sub_figure05_37.csv","sub_figure05_73.csv","sub_figure05_109.csv","sub_figure05_154.csv","sub_figure05_211.csv","sub_figure05_240.csv",\
#            "sub_figure06_63.csv","sub_figure06_99.csv","sub_figure06_159.csv",\
#            "sub_figure08_70.csv","sub_figure08_96.csv","sub_figure08_139.csv",\
#            "sub_figure09_83.csv","sub_figure09_128.csv",\
            "sub_figure09.csv"]
#compute_correlation(file_paths)

batch_size=32
###确定是否使用投票法进行预测
vote_mode=False
#load_model_list=["figure_weights/figure01_100.hdf5","figure_weights/figure02_101.hdf5","figure_weights/figure04_191.hdf5",\
#                 "figure_weights/figure04_350.hdf5"]
load_model_list=[opj("figure_weights",f[4:-4]+".hdf5") for f in file_paths]                 
#preprocess_modes=["01","02","04","04"]
preprocess_modes=[f[10:12] for f in file_paths]
#test_logloss=[0.16,0.1748,0.1714,0.1852]##test log loss
#models_weights=turn_logloss_to_modelweights(test_logloss)##turn test log loss into weights
models_weights=None
ble_log=True##如果为true时，models_weights必须为None
ble_mulx=True##默认为false（一个平均的preds），为true时，多维逻辑回归
vote_sub_file="sub_vote14.csv"

###不使用投票法进行预测
##确定是否训练新的模型
preprocess_mode="01"
show_pics=False
training_mode=False
train_load_model=False
train_load_path="figure_weights/figure01_test5.hdf5"
train_save_path = "figure_weights/figure01_test5.hdf5"
train_log_path = "figure_weights/figure01_log_test5.csv"

###输出每个模型关于训练数据的预测
output_training_pred=False
##不训练新模型，测试模型
#preprocess_mode="04"
epo=139
test_load_model="figure_weights/figure"+preprocess_mode+"_"+str(epo)+".hdf5"
#test_sub_file="sub_figure"+preprocess_mode+"_"+str(epo)+".csv"
test_sub_file="figure01_test5.csv"

train = pd.read_json("input/train.json")
target_train=train['is_iceberg']
test = pd.read_json("input/test.json")

target_train=train['is_iceberg']
test['inc_angle']=pd.to_numeric(test['inc_angle'], errors='coerce')
train['inc_angle']=pd.to_numeric(train['inc_angle'], errors='coerce')#We have only 133 NAs.
train['inc_angle']=train['inc_angle'].fillna(method='pad')
X_angle=train['inc_angle']
test['inc_angle']=pd.to_numeric(test['inc_angle'], errors='coerce')
X_test_angle=test['inc_angle']

#Generate the training data
X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
if (not vote_mode) and (not output_training_pred):
    X_band_3,X_band_4,X_band_5,X_train=switchPreProcessing(X_band_1,X_band_2,mode=preprocess_mode)
    show_demo_pics(X_band_3,X_band_4,X_band_5,X_train,show=show_pics)

	

X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
if (not vote_mode) and (not output_training_pred):
    X_band_test_3,X_band_test_4,X_band_test_5,X_test=switchPreProcessing(X_band_test_1,X_band_test_2,mode=preprocess_mode)

print("完成加载数据")
print(len(X_band_1))
print(len(X_band_test_1))

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Merge, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger

from keras.applications.vgg16 import VGG16
from keras.layers import concatenate
from sklearn.linear_model import LogisticRegression

# Define the image transformations here
gen = ImageDataGenerator(horizontal_flip = True,
                         vertical_flip = True,
                         width_shift_range = 0.,
                         height_shift_range = 0.,
                         channel_shift_range=0,
                         zoom_range = 0.5,
                         rotation_range = 10)

# Here is the function that merges our two generators
# We use the exact same generator with the same random seed for both the y and angle arrays
def gen_flow_for_two_inputs(X1, X2, y):
    genX1 = gen.flow(X1,y,  batch_size=batch_size,seed=55)
    genX2 = gen.flow(X1,X2, batch_size=batch_size,seed=55)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            #Assert arrays are equal - this was for peace of mind, but slows down training
            #np.testing.assert_array_equal(X1i[0],X2i[0])
            yield [X1i[0], X2i[1]], X1i[1]
#Define binary entropy
def binary_crossentropy(y_true,y_pred):
    return np.mean(-(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred)))
# Finally create generator
def get_callbacks(filepath, patience=2):
#   es = EarlyStopping('val_loss', patience=200, mode="min")
   msave = ModelCheckpoint(filepath, monitor='val_acc', mode='max', save_best_only=True)
   csv=CSVLogger(train_log_path, append=True)
   return [csv, msave]

def getVggAngleModel(X_train):
    input_2 = Input(shape=[1], name="angle")
    angle_layer = Dense(1, )(input_2)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=X_train.shape[1:], classes=1)
    x = base_model.get_layer('block5_pool').output
    x = GlobalMaxPooling2D()(x)
    base_model2 = keras.applications.mobilenet.MobileNet(weights=None, alpha=0.9,input_tensor = base_model.input,include_top=False, input_shape=X_train.shape[1:])

    x2 = base_model2.output
    x2 = GlobalAveragePooling2D()(x2)

    merge_one = concatenate([x, x2, angle_layer])

    merge_one = Dropout(0.6)(merge_one)
    predictions = Dense(1, activation='sigmoid',kernel_initializer='he_normal')(merge_one)
    
    model = Model(input=[base_model.input, input_2], output=predictions)
    
    sgd = Adam(lr=1e-4) #SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    model.summary()
    return model

#No CV with Data Augmentation.
def myAngleCV(X_train, X_angle, X_test,file_path):
    K=5
    folds = list(StratifiedKFold(n_splits=K, shuffle=True, random_state=16).split(X_train, target_train))
    y_test_pred_log = 0
    for j, (train_idx, test_idx) in enumerate(folds):

        print('\n===================FOLD=',j)
        X_train_cv = X_train[train_idx]
        y_train_cv = target_train[train_idx]
        X_holdout = X_train[test_idx]
        Y_holdout= target_train[test_idx]
        
        #Angle
        X_angle_cv=X_angle[train_idx]
        X_angle_hold=X_angle[test_idx]

        #define file path and get callbacks
        
        callbacks = get_callbacks(filepath=file_path, patience=10)
        gen_flow = gen_flow_for_two_inputs(X_train_cv, X_angle_cv, y_train_cv)
        galaxyModel= getVggAngleModel(X_train)
        if train_load_model:
            galaxyModel.load_weights(filepath=train_load_path)
            print("Loading model from:"+train_load_path)
        galaxyModel.fit_generator(
                gen_flow,
                steps_per_epoch=24,
                epochs=500,
                shuffle=True,
                verbose=1,
                validation_data=([X_holdout,X_angle_hold], Y_holdout),
                callbacks=callbacks)

        #Getting the Best Model
        galaxyModel.load_weights(filepath=file_path)
        
        #Getting Training Score
        score = galaxyModel.evaluate([X_train_cv,X_angle_cv], y_train_cv, verbose=0)
#        pred_train=galaxyModel.predict([X_train_cv,X_angle_cv])		
        print('Train loss:', score[0])
        print('Train accuracy:', score[1])
        
        #Getting validation Score.
        score = galaxyModel.evaluate([X_holdout,X_angle_hold], Y_holdout, verbose=0)
#        pred_valid=galaxyModel.predict([X_holdout,X_angle_hold])		
        print('Validation loss:', score[0])
        print('Validation accuracy:', score[1])

        #Getting Train&Validation Scores
        score = galaxyModel.evaluate([X_train,X_angle], target_train, verbose=0)
#        temp_train=galaxyModel.predict([X_train, X_angle])
        print('Train&Validation loss:', score[0])
        print('Train&Validation accuracy:', score[1])
		
        #Predicting Test Scores
        temp_test=galaxyModel.predict([X_test, X_test_angle])
        y_test_pred_log+=temp_test.reshape(temp_test.shape[0])
        break
    y_test_pred_log=y_test_pred_log

    return y_test_pred_log
    
def predict(X_test,X_test_angle,load_model):
    print("Load model from:"+load_model)
    galaxyModel= getVggAngleModel(X_train)
    galaxyModel.load_weights(filepath=load_model)
    score = galaxyModel.evaluate([X_train,X_angle], target_train, verbose=0)
    print('训练集所有 loss:', score[0])
    print('训练集所有 accuracy:', score[1])    
    temp_test=galaxyModel.predict([X_test, X_test_angle])
    return temp_test.reshape(temp_test.shape[0])
    
##多个模型投票，可以设置每个模型的权重
def merge_model_eval(load_model_list,preprocess_modes,model_weights=None):
    train_idx=0
    valid_idx=0
    y_test_pred=0
    y_train_pred=0
    y_valid_pred=0
    y_train_all_pred=0
    ble_train_preds_in=np.zeros((len(X_band_1),len(load_model_list)))
    ble_test_preds_in=np.zeros((len(X_band_test_1),len(load_model_list)))
    for j,model_path in enumerate(load_model_list):
        print("\n")
        pre_mode=preprocess_modes[j]
        X_band_3,X_band_4,X_band_5,X_train_=switchPreProcessing(X_band_1,X_band_2,mode=pre_mode)
        X_band_test_3,X_band_test_4,X_band_test_5,X_test_=switchPreProcessing(X_band_test_1,X_band_test_2,mode=pre_mode)
        if j==0:
            folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=16).split(X_train_, target_train))
            train_idx, valid_idx=folds[0];
        
        X_train_cv = X_train_[train_idx];y_train_cv = target_train[train_idx]
        X_holdout = X_train_[valid_idx];Y_holdout= target_train[valid_idx]
        X_angle_cv=X_angle[train_idx];X_angle_hold=X_angle[valid_idx]
    
        galaxyModel= getVggAngleModel(X_train_)
        galaxyModel.load_weights(filepath=model_path)
        
        score = galaxyModel.evaluate([X_train_cv,X_angle_cv], y_train_cv, verbose=0)
        temp_train=galaxyModel.predict([X_train_cv,X_angle_cv])
        print('Train loss:', score[0])
        print('Train accuracy:', score[1])
        
        #Getting validation Score.
        score = galaxyModel.evaluate([X_holdout,X_angle_hold], Y_holdout, verbose=0)
        temp_valid=galaxyModel.predict([X_holdout,X_angle_hold])
        print('Validation loss:', score[0])
        print('Validation accuracy:', score[1])

        #Getting Train&Validation Scores
        score = galaxyModel.evaluate([X_train_,X_angle], target_train, verbose=0)
        temp_train_all=galaxyModel.predict([X_train_,X_angle])
        print('Train&Validation loss:', score[0])
        print('Train&Validation accuracy:', score[1])
        
        temp_test=galaxyModel.predict([X_test_, X_test_angle])
        if ble_mulx:
            ble_train_preds_in[:,j]=temp_train_all[:,0]
            ble_test_preds_in[:,j]=temp_test[:,0]
        else:
            if model_weights!=None:
                y_test_pred+=model_weights[j]*temp_test.reshape(temp_test.shape[0])  
                y_train_pred+=model_weights[j]*temp_train.reshape(temp_train.shape[0])
                y_valid_pred+=model_weights[j]*temp_valid.reshape(temp_valid.shape[0])
                y_train_all_pred+=model_weights[j]*temp_train_all.reshape(temp_train_all.shape[0])
            else:
                y_test_pred+=(1./len(load_model_list))*temp_test.reshape(temp_test.shape[0])  
                y_train_pred+=(1./len(load_model_list))*temp_train.reshape(temp_train.shape[0])
                y_valid_pred+=(1./len(load_model_list))*temp_valid.reshape(temp_valid.shape[0])
                y_train_all_pred+=(1./len(load_model_list))*temp_train_all.reshape(temp_train_all.shape[0])            
    print("Train bce loss:"+str(binary_crossentropy(y_train_cv,y_train_pred)))
    print("Valid bce loss:"+str(binary_crossentropy(Y_holdout,y_valid_pred)))
    print("Train-all bce loss:"+str(binary_crossentropy(target_train,y_train_all_pred)))
    
    y_test_sub=0
    ##Add Blending: Logistic Regression
    if ble_mulx and ble_log:
        clf=LogisticRegression()
        clf.fit(ble_train_preds_in,target_train)
        print(ble_train_preds_in[0])
        print(ble_test_preds_in[0])
        y_test_temp=clf.predict_proba(ble_test_preds_in)[:,1]
        y_test_sub=y_test_temp
    elif ble_log:
        clf=LogisticRegression()
        temp1=np.zeros((len(y_train_all_pred),1))
        temp2=np.zeros((len(y_test_pred),1))
        for i in range(len(y_train_all_pred)):
            temp1[i,0]=y_train_all_pred[i]
        for i in range(len(y_test_pred)):
            temp2[i,0]=y_test_pred[i]
        print (temp1[0])
        print (temp2[0])
        clf.fit(temp1,target_train)
        y_test_temp=clf.predict_proba(temp2)[:,1]
        y_test_temp=(y_test_temp-y_test_temp.min())/(y_test_temp.max()-y_test_temp.min())
        y_test_sub=y_test_temp
    else:
        y_test_sub=y_test_pred
    return y_test_sub  
if vote_mode:
    preds=merge_model_eval(load_model_list,preprocess_modes,model_weights=models_weights)
    submission = pd.DataFrame()
    submission['id']=test['id']
    submission['is_iceberg']=preds
    submission.to_csv(vote_sub_file, index=False)
elif training_mode:    
    preds=myAngleCV(X_train, X_angle, X_test,train_save_path)
elif output_training_pred:
    for j,model_path in enumerate(load_model_list):
        print("\n")
        pre_mode=preprocess_modes[j]
        X_band_3,X_band_4,X_band_5,X_train_=switchPreProcessing(X_band_1,X_band_2,mode=pre_mode)
        galaxyModel= getVggAngleModel(X_train_)
        galaxyModel.load_weights(filepath=model_path)
        print("Loading model from "+str(model_path))
        temp_train_all=galaxyModel.predict([X_train_,X_angle])
        temp_train_all.reshape(temp_train_all.shape[0])
        submission = pd.DataFrame()
        submission['id']=train['id']
        submission['is_iceberg']=temp_train_all
        submission.to_csv(opj("training_predictions","training_"+file_paths[j]), index=False)        

else:
    
    preds=predict(X_test,X_test_angle,test_load_model)
    submission = pd.DataFrame()
    submission['id']=test['id']
    submission['is_iceberg']=preds
submission.to_csv(test_sub_file, index=False)
