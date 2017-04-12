from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class ERPData(object):
    def __init__(self,subject,session,eeg,labels,stimuli,channels):
        """
        Object to contain the ERP data.
        This is used as an intermediate representation that can be processed and fed into the decoders easily.

        Parameters
        ----------
        subject : string
                the name of the the subject

        session : string
                the name of the session

        eeg : a list of numpy arrays
                each component of the list is an array of (S x C x L)
                S is the number of stimuli in the trial, C i the number of EEG channels and L is the length of the trial

        labels : a list of numpy arrays
                one array per trial, with the target -non target labelling per stimulus in the trial

        stimuli : a list of numpy arrays
                For each trial it has the structure of (S, #S).
                S is the number of stimulus presentations.
                #S is the number of stimuli presented in a stimulus presentation (e.g. std row-column: 6, Amuse: 1)
                If stimulus type j is presented in stimulus i of trial t then stimuli[t][i,:] contains j
                If the number of stimuli differs per presentation, -1 can be used as padding

        channels : An iterable containing the channel assignations

        Returns
        -------
        An object with the attributes described above and extended with
        target_stimulus : a list containing for each trial which is the target stimulus
        """
        self.subject = subject
        self.session = session
        self.eeg = eeg
        self.labels = labels
        self.stimuli = stimuli
        self.channels = channels

        # Figure out which stimulus is the target.
        # The target stimulus is the one where its binary intensification mask equals the label of the trial
        self.target_stimulus = []
        # Make it O(N * #S * #P) with #P the highlighted per stimulus
        num_stimuli = stimuli.max()+1
        for stim,label in zip(self.stimuli,self.labels):
            self.target_stimulus.append((np.array([[idx in s for s in stim] for idx in range(num_stimuli)])==label[np.newaxis]).sum(axis=1).argmax())
        self.target_stimulus = np.hstack(self.target_stimulus)

    def get_data_as_xy(self):
        """
        Get a representation of the data for a traditional supervised machine learning setting.

        Returns
        -------
        X : numpy array of (N,C,L) where N is the number of stimuli presented, C is the number of channels and L is the number of samples per channel
        Y : is an array of (N,) where N is the number of stimuli presented

        Remark: a stimulus is equal to a datapoint
        """
        return np.vstack(self.eeg), np.hstack(self.labels)

class TrialIterator(object):
    """
    Iterator that iterates over trials
    It returns X, U
    X are the processed datapoints in this trial as (S, C, T) with S=Stimuli, C=Channels , T=Time
    U are the stimuli that are highlighted in a trial

    It does not send label information
    """
    def __init__(self,data):
        assert isinstance(data,ERPData)
        self.data = data
        self.trial_idx = 0

    def __iter__(self):
        return self

    def next(self):
        """
        Iterate over all the stimulus presentations
        :return:
        """
        if self.trial_idx == self.data.eeg.shape[0]:
            raise StopIteration
        else:
            x, s = self.data.eeg[self.trial_idx], self.data.stimuli[self.trial_idx]
            self.trial_idx += 1
            return x, s
