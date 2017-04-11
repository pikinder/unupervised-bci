from scipy.signal import butter,lfilter
import numpy as np


def flatten_normalise_bias(x):
    x = x.reshape(x.shape[0],-1)
    x = x - np.mean(x,axis=1)[:,np.newaxis]
    x = x/(np.var(x,axis=1)+0.000001)[:,np.newaxis]
    x = np.hstack([x,np.ones((x.shape[0],1))])
    return x

def causal_filter(x,low,high,fs):
    """
    Apply a causal bandpass filter to the EEG...
    :param x: eeg data (channel x timesteps)
    :param low: low cutoff
    :param high: high cutoff
    :param fs: sampling frequency
    :return: the frequency sampled EEG
    """
    (b,a)=butter(4,(2.0*low/fs, 2.0*high/fs),btype='band')
    return lfilter(b,a,x,axis=1)