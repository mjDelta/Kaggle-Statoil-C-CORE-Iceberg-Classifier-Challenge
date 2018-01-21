# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 15:42:58 2018

@author: ZMJ
"""
from keras.layers import Conv2D,MaxPooling2D,Dense,Merge,Flatten,Dropout
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint,CSVLogger
import pandas as pd

def get_callbacks(filepath, train_log_path):
   msave = ModelCheckpoint(filepath, monitor='val_acc', mode='max', save_best_only=True)
   csv=CSVLogger(train_log_path, append=True)
   return [csv, msave]
 
def vgg16light_extract_features(x_train,##bands
                                y_train,##is_iceberg
                                x_angle,##inc_angle
                                file_path,##model save path
                                train_log_path,##model log path
                                train_load_path,## load model path
                                X_test,##bands test
                                X_test_angle,## bands inc_angle test
                                test_sub_file,## test sub file path
                                mode="training"
                                ):
    model=Sequential()
#    conv_base=VGG16(weights="imagenet",include_top=False,input_shape=(75,75,3))
#    conv_base.summary()
    conv_base=Sequential()
    channel_=0.5
    conv_base.add(Conv2D(int(16*channel_),(3,3),activation="relu",padding="same",input_shape=(75,75,3)))
    conv_base.add(Conv2D(int(16*channel_),(3,3),activation="relu",padding="same"))
    conv_base.add(MaxPooling2D((2,2)))
    conv_base.add(Conv2D(int(32*channel_),(3,3),activation="relu",padding="same"))
    conv_base.add(Conv2D(int(32*channel_),(3,3),activation="relu",padding="same"))
    conv_base.add(MaxPooling2D((2,2)))    
    conv_base.add(Conv2D(int(64*channel_),(3,3),activation="relu",padding="same"))
    conv_base.add(Conv2D(int(64*channel_),(3,3),activation="relu",padding="same"))
    conv_base.add(Conv2D(int(64*channel_),(3,3),activation="relu",padding="same"))
#    conv_base.add(Dropout(0.2))

    conv_base.add(MaxPooling2D((2,2)))   
    conv_base.add(Conv2D(int(64*channel_),(3,3),activation="relu",padding="same"))
    conv_base.add(Conv2D(int(64*channel_),(3,3),activation="relu",padding="same"))
    conv_base.add(Conv2D(int(64*channel_),(3,3),activation="relu",padding="same"))
#    conv_base.add(Dropout(0.2))

    conv_base.add(MaxPooling2D((2,2)))   
    conv_base.add(Conv2D(int(64*channel_),(3,3),activation="relu",padding="same"))
    conv_base.add(Conv2D(int(64*channel_),(3,3),activation="relu",padding="same"))
    conv_base.add(Conv2D(int(64*channel_),(3,3),activation="relu",padding="same"))
#    conv_base.add(Dropout(0.2))

    conv_base.add(MaxPooling2D((2,2)))   
    conv_base.add(Flatten())
    
    angle_base=Sequential()
    angle_base.add(Dense(32,input_shape=(1,),activation="relu"))
#    angle_base = Dense(1,input_shape=(1,))
    model.add(Merge([conv_base,angle_base], mode='concat'))
    
    model.add(Dense(int(64*channel_)))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation="sigmoid"))
    
    modeltwo=Model([conv_base.input,angle_base.input],model.output)

    modeltwo.compile(optimizer="rmsprop",
                      loss="binary_crossentropy",
                      metrics=["acc"])
    modeltwo.summary()
    callbacks = get_callbacks(filepath=file_path,train_log_path)

    if mode=="training":
      history=modeltwo.fit([x_train,x_angle],y_train,
                        epochs=500,
                        batch_size=128,
                        callbacks=callbacks,
                        validation_split=0.2)
    else: 
      modeltwo.load_weights(train_load_path)
      print("Loading weights from "+train_load_path)
      loss,acc=modeltwo.evaluate([x_train,x_angle],y_train)
      print("Loss of training data:"+str(loss))
      print("Acc of training data:"+str(acc))
      test_preds=modeltwo.predict([X_test, X_test_angle]).reshape(len(X_test))
      submission = pd.DataFrame()
      submission['id']=test['id']
      submission['is_iceberg']=test_preds
      submission.to_csv(test_sub_file, index=False)
if __name__=="__main__":
  train_log_path="figure_weights/figure01_log_test5.csv"
  train_load_path="figure_weights/figure01_test5.hdf5"
  train_save_path = "figure_weights/figure01_test5.hdf5"
  
  preprocess_mode="01"
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
  X_band_3,X_band_4,X_band_5,X_train=switchPreProcessing(X_band_1,X_band_2,mode=preprocess_mode)
  X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
  X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])  
  X_band_test_3,X_band_test_4,X_band_test_5,X_test=switchPreProcessing(X_band_test_1,X_band_test_2,mode=preprocess_mode)
  vgg16light_extract_features(X_train,
                              target_train,
                              X_angle,
                              train_save_path,
                              train_log_path,
                              train_load_path,
                              X_test,
                              X_test_angle,
                              test_sub_file,
                              training)