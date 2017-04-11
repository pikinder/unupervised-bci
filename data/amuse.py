from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import config

from scipy.io import loadmat
from tools.preprocess import causal_filter
from tools.fileio import store
from data.erp_data import ERPData

# Channels that should not be used for EEG
_exclude_channels = ['EOGv','EOGh','MasL','MasR']

class AmuseMat(object):
    """
    Struct-like object to work with amuse data
    """
    def __init__(self,subject,session):
        """
        Load a matfile from disk and transform it into the AmuseMat object

        :param subject: matfile that will be loaded with scipy.io.loadmat
        :param session: 'calib' or 'online'
        :return: object that contains the relevant content for the matfile
        """
        assert session in ['calib','online']

        # Which part of the struct to load?
        if session == 'calib':
            idx = 0
        else:
            idx = 1

        # Load the data
        mf = loadmat('%s/%s'%(config._raw,subject))

        # Keep track of all channels in the data
        tmp_channels = [c[0] for c in mf['mnt']['clab'][0][0][0]]

        self.subject = subject
        self.session = session

        # Only retain some channels
        self.channels = [c for c in tmp_channels if c not in _exclude_channels]

        # The EEG (retained channels x time)
        self.eeg =  np.vstack([x for (x,c) in zip(mf['data'][0][idx]['X'][0][0].T,tmp_channels) if c not in _exclude_channels]) #
        assert self.eeg.shape[0] == 58 # Check that we always have the same number of channels...:

        # Sampling frequency
        self.fs= 1.0 * mf['data'][0][idx]['fs'][0][0][0][0]
        assert self.fs == 250 # make sure the data for amuse is uniform :)

        # Number of different stimuli
        self.Ns = 6

        # index of the stimulus in the EEG array
        self.idx = mf['data'][0][idx]['trial'][0][0][0]

        # Labels, target non-target
        self.label = 1.0*(mf['data'][0][idx]['y'][0][0][0]==1)

        # Presented symbol
        self.stimulus = mf['data'][0][idx]['y_stim'][0][0][0].squeeze()-1

        # trial identifier
        self.trial = mf['data'][0][idx]['y_trialIdx'][0][0][0].squeeze()

def preprocess_amuse_mat(subject):
    """
    Transform an AmuseMat object into a list with 2  ERPData Objects.
    Save it to disk.

    """
    _bp_low = .5
    _bp_high = 15.
    _subsample = 10 # Assume 250 Hz -> 25 Hz, below nyquist, but not a problem for BCI ...
    _fs_needed = 250
    _max_time = 0.7 # Time after the stimulus
    _offsets =  np.r_[0:int(_fs_needed*_max_time):_subsample]
    _num_stimuli = 6 # Hardcode this

    matfiles  = [AmuseMat(subject,session) for session in ['calib','online']]
    data = []

    for mf in matfiles:
        # Hardcoded the sampling frequency for this dataset
        assert mf.fs == _fs_needed
        # Filter the EEG
        mf.eeg = causal_filter(mf.eeg,_bp_low,_bp_high,mf.fs)

        prev_trial = -1
        x = []
        y = []
        stim = []
        # The trials count from 1 in the mat file!
        for trial, stimulus, label, idx in zip(mf.trial,mf.stimulus,mf.label,mf.idx):
            assert trial >= prev_trial
            if trial != prev_trial:
                x.append([])
                y.append([])
                stim.append([])
                prev_trial = trial

            stim[-1].append([stimulus])
            y[-1].append(label)
            x[-1].append(mf.eeg[:,idx+_offsets])

        data.append(ERPData(
            subject = mf.subject,
            session = mf.session,
            eeg = np.concatenate([[xx]for xx in x],axis=0), # Weird line of code. But creates a shape of (trials, stimuli, channels, time)
            labels = np.concatenate([[yy] for yy in y],axis=0),
            stimuli = np.concatenate([[np.vstack(ss)] for ss in stim],axis=0),
            channels = mf.channels
        ))
    store(data,('%s/amuse_%s.pkl')%(config._processed,subject))
