# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 15:39:49 2018

@author: ZMJ
"""
import numpy as np
import time
from scipy.ndimage import gaussian_filter
from skimage.morphology import reconstruction
##ROF去噪
def denoise_rof(im, U_init, tolerance=0.1, tau=0.125, tv_weight=100):
    """ An implementation of the Rudin-Osher-Fatemi (ROF) denoising model
        using the numerical procedure presented in Eq. (11) of A. Chambolle
        (2005). Implemented using periodic boundary conditions 
        (essentially turning the rectangular image domain into a torus!).
    
        Input:
        im - noisy input image (grayscale)
        U_init - initial guess for U
        tv_weight - weight of the TV-regularizing term
        tau - steplength in the Chambolle algorithm
        tolerance - tolerance for determining the stop criterion
    
        Output:
        U - denoised and detextured image (also the primal variable)
        T - texture residual"""
    
    #---Initialization
    m,n = im.shape #size of noisy image

    U = U_init
    Px = im #x-component to the dual field
    Py = im #y-component of the dual field
    error = 1 
    iteration = 0

    #---Main iteration
    while (error > tolerance):
        Uold = U

        #Gradient of primal variable
        LyU = np.vstack((U[1:,:],U[0,:])) #Left translation w.r.t. the y-direction
        LxU = np.hstack((U[:,1:],U.take([0],axis=1))) #Left translation w.r.t. the x-direction

        GradUx = LxU-U #x-component of U's gradient
        GradUy = LyU-U #y-component of U's gradient

        #First we update the dual varible
        PxNew = Px + (tau/tv_weight)*GradUx #Non-normalized update of x-component (dual)
        PyNew = Py + (tau/tv_weight)*GradUy #Non-normalized update of y-component (dual)
        NormNew = np.maximum(1,np.sqrt(PxNew**2+PyNew**2))

        Px = PxNew/NormNew #Update of x-component (dual)
        Py = PyNew/NormNew #Update of y-component (dual)

        #Then we update the primal variable
        RxPx =np.hstack((Px.take([-1],axis=1),Px[:,0:-1])) #Right x-translation of x-component
        RyPy = np.vstack((Py[-1,:],Py[0:-1,:])) #Right y-translation of y-component
        DivP = (Px-RxPx)+(Py-RyPy) #Divergence of the dual field.
        U = im + tv_weight*DivP #Update of the primal variable

        #Update of error-measure
        error = np.linalg.norm(U-Uold)/np.sqrt(n*m);
        iteration += 1;

    #The texture residual
#    T = im - U
    
    return U
##形态学
def transform(img):

    img=gaussian_filter(img,2.5)
    seed=np.copy(img)
    seed[1:-1,1:-1]=img.min()
    mask=img
    ##morphology processing
    pro_img=reconstruction(seed,mask,method="dilation")
    return img-pro_img
def switchPreProcessing(band1,band2,mode="01"):
    start=time.time()
    if mode=="01":        

        X_band_3=np.array([(band-np.min(band))/(np.max(band)-np.min(band)) for band in band1])
        X_band_4=np.array([(band-np.min(band))/(np.max(band)-np.min(band)) for band in band2])
        X_band_5=np.fabs(np.subtract(X_band_3,X_band_4))
    elif mode=="02":
        X_band_3=np.fabs(np.subtract(band1,band2))
        X_band_4=np.maximum(band1,band2)
        X_band_5=np.minimum(band1,band2)
    elif mode=="03":
        x_band_temp=band1+band2
        X_band_3=np.array([(band-np.mean(band)/(np.max(band)-np.min(band))) for band in band1])
        X_band_4=np.array([(band-np.mean(band)/(np.max(band)-np.min(band))) for band in band2])
        X_band_5=np.array([(band-np.mean(band)/(np.max(band)-np.min(band))) for band in x_band_temp])
    elif mode=="04":
        x_band_temp=band1+band2
        
        X_band_3=np.array([(band-np.min(band)/(np.max(band)-np.min(band))) for band in band1])
        X_band_4=np.array([(band-np.min(band)/(np.max(band)-np.min(band))) for band in band2])
        X_band_5=np.array([(band-np.min(band)/(np.max(band)-np.min(band))) for band in x_band_temp])
    elif mode=="05":##傅里叶变换
        x_band_temp=band1+band2
        X_band_3=np.array([np.abs(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2((band-np.min(band))/(np.max(band)-np.min(band))))))) for band in band1])
        X_band_4=np.array([np.abs(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2((band-np.min(band))/(np.max(band)-np.min(band))))))) for band in band2])
        X_band_5=np.array([np.abs(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2((band-np.min(band))/(np.max(band)-np.min(band))))))) for band in x_band_temp])
    elif mode=="06":##ROF去噪
        x_band_temp=band1+band2
        X_band_3=np.array([denoise_rof((band-np.min(band))/(np.max(band)-np.min(band)),(band-np.min(band))/(np.max(band)-np.min(band))) for band in band1])
        X_band_4=np.array([denoise_rof((band-np.min(band))/(np.max(band)-np.min(band)),(band-np.min(band))/(np.max(band)-np.min(band))) for band in band2])
        X_band_5=np.array([denoise_rof((band-np.min(band))/(np.max(band)-np.min(band)),(band-np.min(band))/(np.max(band)-np.min(band))) for band in x_band_temp])      
    elif mode=="07":#先ROF再傅里叶变换
        x_band_temp=band1+band2
        X_band_3_temp=np.array([denoise_rof((band-np.min(band))/(np.max(band)-np.min(band)),(band-np.min(band))/(np.max(band)-np.min(band))) for band in band1])
        X_band_4_temp=np.array([denoise_rof((band-np.min(band))/(np.max(band)-np.min(band)),(band-np.min(band))/(np.max(band)-np.min(band))) for band in band2])
        X_band_5_temp=np.array([denoise_rof((band-np.min(band))/(np.max(band)-np.min(band)),(band-np.min(band))/(np.max(band)-np.min(band))) for band in x_band_temp])      

        X_band_3=np.array([np.abs(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2((band-np.min(band))/(np.max(band)-np.min(band))))))) for band in X_band_3_temp])
        X_band_4=np.array([np.abs(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2((band-np.min(band))/(np.max(band)-np.min(band))))))) for band in X_band_4_temp])
        X_band_5=np.array([np.abs(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2((band-np.min(band))/(np.max(band)-np.min(band))))))) for band in X_band_5_temp])   
    elif mode=="08":#先傅里叶变换再ROF
        x_band_temp=band1+band2
        X_band_3_temp=np.array([np.abs(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2((band-np.min(band))/(np.max(band)-np.min(band))))))) for band in band1])
        X_band_4_temp=np.array([np.abs(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2((band-np.min(band))/(np.max(band)-np.min(band))))))) for band in band2])
        X_band_5_temp=np.array([np.abs(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2((band-np.min(band))/(np.max(band)-np.min(band))))))) for band in x_band_temp])        
        
        X_band_3=np.array([denoise_rof((band-np.min(band))/(np.max(band)-np.min(band)),(band-np.min(band))/(np.max(band)-np.min(band))) for band in X_band_3_temp])
        X_band_4=np.array([denoise_rof((band-np.min(band))/(np.max(band)-np.min(band)),(band-np.min(band))/(np.max(band)-np.min(band))) for band in X_band_4_temp])
        X_band_5=np.array([denoise_rof((band-np.min(band))/(np.max(band)-np.min(band)),(band-np.min(band))/(np.max(band)-np.min(band))) for band in X_band_5_temp])   
    elif mode=="09":#形态学变换
        x_band_temp=band1+band2
        
        X_band_3_temp=np.array([transform(band) for band in band1])
        X_band_4_temp=np.array([transform(band) for band in band2])
        X_band_5_temp=np.array([transform(band) for band in x_band_temp])      
        
        X_band_3=np.array([(band-np.min(band))/(np.max(band)-np.min(band)) for band in X_band_3_temp])
        X_band_4=np.array([(band-np.min(band))/(np.max(band)-np.min(band)) for band in X_band_4_temp])
        X_band_5=np.array([(band-np.min(band))/(np.max(band)-np.min(band)) for band in X_band_5_temp])
    print("Preprocessing mode : "+mode+" Prepare time:"+str(time.time()-start)+"s")
    X_train_ = np.concatenate([X_band_3[:, :, :, np.newaxis],X_band_4[:, :, :, np.newaxis],X_band_5[:, :, :, np.newaxis]], axis=-1)
    return X_band_3,X_band_4,X_band_5,X_train_
