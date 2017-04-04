from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.io import loadmat
import config
# Channels that should not be used for EEG
_exclude_channels = ['EOGv','EOGh','MasL','MasR']


class AmuseMat(object):
    """
    Struct-like object to work with amuse data
    """
    def __init__(self,subject,session):
        """
        Transform a loaded matfile from the amuse dataset into an object where everything.

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

        # Number of different stimuli
        self.Ns = 6

        # index of the stimulus in the EEG array
        self.idx = mf['bbci_mrk'][0][idx]['time'][0][0][0]

        # Labels, target non-target
        self.label = mf['bbci_mrk'][0][idx]['y'][0][0][0]

        # Presented symbol
        self.stimulus = mf['bbci_mrk'][0][idx]['event'][0][0]['desc'][0][0]

        # trial identifier
        self.trial = mf['bbci_mrk'][0][idx]['event'][0][0]['trial_idx'][0][0]



