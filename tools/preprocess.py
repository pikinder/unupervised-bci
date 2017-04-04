from scipy.signal import butter,lfilter

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