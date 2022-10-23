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
    origin_df = pd.read_csv(dataset_filname)
    new_df = origin_df.dropna()

    return np.mean(new_df[['crime', 'police', 'enroll']], axis=0)

def extract_estimator(dataset_filname: str):
    """Calculates the ols estimator vector

    Parameters
    ==========
    dataset_filename: str; name of the dataset where the data is extracted from

    Returns
    =======
    OLS estimator vector
    """
    origin_df = pd.read_csv(dataset_filname)
    new_df = origin_df.dropna()
    new_df['beta0'] = 1
    log_enroll = np.array(new_df[['lenroll']])
    log_crime = np.array(new_df['lcrime'])
    new_log_enroll = np.append(
        log_enroll.reshape(-1, 1), new_df['beta0'].to_numpy().reshape(-1, 1), axis=1)
    betahat = np.dot(np.linalg.inv(np.dot(np.transpose(
        new_log_enroll), new_log_enroll)), np.dot(np.transpose(new_log_enroll), log_crime))

    return betahat
