"""
   Use to create your own functions for reuse 
   across the assignment

   Inside part_1_template_solution.py, 
  
     import new_utils
  
    or
  
     import new_utils as nu
"""
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
def scale_data(alp):
  scaled_x=alp.astype('float32')/255.0
  return scaled_x

def scale(alp):
    if np.max(alp)<=1 and np.min(alp)>=0:
      return True
    else:
      return False
    
def checklabels(y):
    val=np.unique(y)
    for i in val:
       if type(i)=='str':
            return 'String Type'  
    else:
        return 'Integers'
def accuracy(cm):
   return np.diagonal(cm).sum() / np.sum(cm)

def remove_90_9s(X: NDArray[np.floating], y: NDArray[np.int32]):
    """
    Filter the dataset to include only the digits 7 and 9.
    Parameters:
        X: Data matrix
        y: Labels
    Returns:
        Filtered data matrix and labels
    Notes:
        np.int32 is a type with a range based on 32-bit ints
        np.int has no bound; it can hold arbitrarily long numbers
    """
  
    nine_idx = (y == 9)
    #print('Ids of nine',nine_idx)
    X_90 = X[nine_idx, :]
    y_90 = y[nine_idx]
    #print(int((X_90.shape[0])*0.1))
    X_90=X_90[:int((X_90.shape[0])*0.1),:]
    y_90=y_90[:int((y_90.shape[0])*0.1)]
    none_nine= (y!=9)
    X_non_9 = X[none_nine, :]
    y_non_9 = y[none_nine]
    fin_X=np.concatenate((X_non_9,X_90),axis=0)
    fin_y=np.concatenate((y_non_9,y_90),axis=0)
    return fin_X, fin_y

def convert_7_0(X: NDArray[np.floating], y: NDArray[np.int32]):
   id_7=(y==7)
   id_0=(y==0)
   y[id_7]=0
  #  try:
  #   X[id_7,:]=X[id_0,:]
  #  except Exception as error:
  #     print(error)
   return X,y

def convert_9_1(X: NDArray[np.floating], y: NDArray[np.int32]):
   id_9=(y==9)
   id_1=(y==1)
   y[id_9]=1
  #  try:
  #   X[id_7,:]=X[id_0,:]
  #  except Exception as error:
  #     print(error)
   return X,y