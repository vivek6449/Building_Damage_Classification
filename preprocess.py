import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, KFold
from lightgbm import LGBMClassifier

import gc

# Importing paths
root_dir = '/home/vivek/Datasets/'
project_dir = 'Building_Damage_Classification/'
file_path = os.path.join(root_dir,project_dir)

buil_owner_path = os.path.join(file_path,"Building_Ownership_Use.csv")
buil_stru_path = os.path.join(file_path, "Building_Structure.csv")
train_path = os.path.join(file_path, "train.csv")
test_path = os.path.join(file_path, "test.csv")

def building_model_input():
    print("Reading Building Ownership file.")
    buil_owner=pd.read_csv(buil_owner_path)
    
    print("Building Ownership shape: ", buil_owner.shape)
    
    print("transform to dummies")
    buil_owner = pd.concat([buil_owner, pd.get_dummies(buil_owner.legal_ownership_status, prefix='owner')], axis=1).drop('legal_ownership_status', axis=1)
    
    print("Reading building structural file")
    buil_stru = pd.read_csv(buil_stru_path)

    print("Building structure shape: ", buil_stru.shape)
    
    print("transform to dummies")
    
    buil_land_dum = pd.get_dummies(buil_stru.land_surface_condition, prefix='land_')
    buil_foun = pd.get_dummies(buil_stru.foundation_type, prefix='foun')
    buil_roof = pd.get_dummies(buil_stru.roof_type, prefix='roof')
    buil_grnd = pd.get_dummies(buil_stru.ground_floor_type, prefix='grnd')
    buil_oth = pd.get_dummies(buil_stru.other_floor_type, prefix='oth')
    buil_pos = pd.get_dummies(buil_stru.position, prefix='pos')
    buil_plan = pd.get_dummies(buil_stru.plan_configuration, prefix='plan')
    buil_con = pd.get_dummies(buil_stru.condition_post_eq, prefix='con')

    buil_stru = pd.concat([buil_stru, buil_land_dum, buil_foun, buil_roof, buil_grnd, buil_oth, buil_pos, buil_pos,buil_plan, buil_con],axis=1).drop(['land_surface_condition','foundation_type','roof_type','ground_floor_type','other_floor_type','position','plan_configuration','condition_post_eq'], axis=1)
    
    del buil_land_dum, buil_foun, buil_roof, buil_grnd, buil_oth, buil_pos,buil_plan, buil_con
    
    gc.collect()

    print("Read train and test dataset")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    print("Training data shape: ", train.shape)
    print("Testing data shape: ", test.shape)
    
    
