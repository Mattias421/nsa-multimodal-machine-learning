"""
A set of NEGATIVE SELECTION ALGORITHM (NSA) related functions, relies on good file structure
Mattias Cross
"""
from sklearn.decomposition import PCA
import numpy as np
import argparse
import math
import os 
import sys
import pickle
import csv

##################
# some functions #
##################

modalities = ['sw_l_acc', 'sw_l_gyr', 'sw_r_acc', 'sw_r_gyr', 'sp_r_acc', 'sp_r_gyr', 'eb_l_acc', 'eb_l_gyr']   

def save_dict(I):
    """remove redundant detectors then save as np array"""
    for m in modalities:
        Ir = I[m][0]
        R = np.array([])
        np.append(R,I[m][0])
        np.save("R_"+m, R)
        #print(np.array(I[m][0]).shape)
        np.save("BS_"+m, np.array(I[m][1]))

def load_dict():
    I = {}
    for m in modalities:
        I[m] = list(np.load("I_"+m+".npy",allow_pickle=True))
    return I

def minkowski(a, b, l=2):
    """minkowski distance in nd"""
    s = 0

    for i in range(len(a)-2):
        s += np.power((a[i] - b[i]),l)
    return np.power(s, 1/l)

def getData(path, modality, hp=0, lp=100000):
    """convert all mm-fit workout data (from 1 modality) to np array""" 
    raw_data = [] 
    dataType = '_'+modality+'.npy'
    _,dirnames,_ = next(os.walk(path))
    for d in dirnames:
        data = np.load(os.path.join(path,d,d + dataType))
        x = data[:, (0, 2, 3, 4)][:,1]
        y = data[:, (0, 2, 3, 4)][:,2]
        z = data[:, (0, 2, 3, 4)][:,3]
        mag = np.square(x) + np.square(y) + np.square(z)
        mag = np.sqrt(mag)
        raw_data.append(np.array(mag)[hp:lp])

    return raw_data

def prepareData(data,hp=0, lp=100000):
    """prepare data for detection"""
    x = data[:, (0, 2, 3, 4)][:,1]
    y = data[:, (0, 2, 3, 4)][:,2]
    z = data[:, (0, 2, 3, 4)][:,3]
    mag = np.square(x) + np.square(y) + np.square(z)
    mag = np.sqrt(mag)

    return [np.array(mag)[hp:lp]]

def generateR(S,nR0,m):
    """generate set of detectors and bounding sphere"""
    centre = np.zeros(len(S[0][1]))
    rC = np.max([minkowski(c[1], centre) for c in S]) * 2              #range of centre is based on point of self furthest away from the mean
    BS = np.array([rC])                                                  #bounding sphere
    C = np.random.uniform(low=-rC,high=rC,size=(nR0,m))
    rR = abs(rC) / 2                                                   #range of radius
    R = [[np.random.randint(0,rR),c] for c in C]           #creates n-spheres stored as [radius,centre]
    
    return [R, BS]

def kissingR(R, S):
    """update radius of detectors so it -kisses- nearest self"""
    for i in range(len(R)):
        Sn = S[np.argmin([minkowski(R[i][1],s[1]) for s in S])] #nearest self
        R[i][0] = minkowski(R[i][1],Sn[1]) - Sn[0] - 10 #k = 10
        
    return R

def censor(R, S, BS, m):
    """centered censoring (very filling/ overfitting)"""
    nR = range(len(R))
    censoring = True
    while censoring:       
        censoring = False
        R = kissingR(R, S)
        #visualise(R, S, False)
        for i in nR:
            r = R[i]
            for s in S:
                if minkowski(r[1],s[1]) < r[0] + s[0]:
                    #covering self - could decrease radius as penalty
                    #move the detector by distance r 
                    c = R[i,1] + np.random.uniform(low=-r[0],high=r[0],size=m)
                    R[i] = [r[0],c]
                    censoring = True                   
                elif minkowski(r[1],np.zeros(m)) > BS[0]:
                    #return sphere to centre if it is out of bounds (fills alot more space)
                    c = np.zeros(len(r[1]))
                    R[i] = [r[0],c]
                    censoring = True  
    for i in nR:
        if R[i][1].all() == np.zeros(m).all():
            R[i][0] = 0
    return R

#####################
### THE ALGORITHM ###
#####################

def generateI(modality, rS=100, nR0=100, m=4):
    """generate an immune system for a given modality"""
    
    train_data = getData('mm-fit/', modality) #data assumed to be in mm-fit
    pc = PCA(n_components=m)

    S = [[rS, c] for c in pc.fit_transform(train_data)] #generate selfs with radius rS

    RBS = generateR(S, nR0, m) #generate set of detectors R and its boundary sphere BS  

    RBS[0] = censor(RBS[0], S, RBS[1], m) #sensor the detectors so that they don't cover self
    
    RBS.append(pc) #save pca for later
    
    return RBS

def generateNI(rS=100,nR0=100,m=4):
    """generates an immune system for each modality
    :param rS: radius of self points
    :param nR0: number of detectors
    :param m: number of dimensions to use
    :return I: a dictionary of immune systems"""
    I = {}                                                                                         

    for modality in modalities:
        I[modality] = generateI(modality)

    return I

def detect(I, D, m):
    """detect noise in data
    :param I: an immune system
    :param D: data to analyse
    :param m: number of dimensions
    :return : true or false"""

    R = I[0]
    BS = I[1]
    pc = I[2]

    d = pc.transform(prepareData(D)) #PCA on data

    if minkowski(d[0], np.zeros(m)) > BS[0]:
        return True
    for i in range(len(R)):
        if minkowski(R[i][1],d[0]) < R[i][0]:
            return True
    return False


