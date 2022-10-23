"""
Justin Chen
Problem Set 3: OLS Exercise
"""
import pandas as pd
import numpy as np

def extract_variable_means(dataset_filname: str):
    """Calculates the mean values of the number of total campus crimes, employed
    police officers, and total college enrollment
    
    Parameters
    ==========
    dataset_filename: str; name of the dataset where the data is extracted from

    Returns
    =======
    mean values of the number of total campus crimes, employed
    police officers, and total college enrollment
    """
    df = pd.read_csv(dataset_filname)
    new_df = df.dropna()
    camp_cri_mean = np.mean(new_df['crime'])
    police_mean = np.mean(new_df['police'])
    college_enroll_mean = np.mean(new_df['enroll'])

    return camp_cri_mean, police_mean, college_enroll_mean

def extract_estimator(dataset_filname: str):
    """Calculates the ols estimator vector
    
    Parameters
    ==========
    dataset_filename: str; name of the dataset where the data is extracted from

    Returns
    =======
    OLS estimator vector 
    """
    df = pd.read_csv(dataset_filname)
    new_df = df.dropna()
    x = np.append(new_df['lenroll'].to_numpy().reshape(-1, 1), new_df['beta0'].to_numpy().reshape(-1, 1), axis=1)
    y = new_df['lcrime']
    betahat = np.dot(np.linalg.inv(np.dot(np.transpose(x) , x)), np.dot(np.transpose(x),y))

    return betahat
