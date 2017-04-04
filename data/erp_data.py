class ERPData(object):
    def __init__(self,subject,session,eeg,labels,stimuli):
        """
        :param subject the subject
        :param session the session
        :param eeg: a list of numpy arrays, one per trial
                    each component of the list is an array of (stimuli x channels x time)
        :param labels: a list of np.array, one per trial, with the target -non target labelling per stimulus un the trial
        :param stimuli: a list of np.array, one per trial with the highlighted stimulus per trial
        :return:
        """
        self.subject = subject
        self.session = session
        self.eeg = eeg
        self.labels = labels
        self.stimuli = stimuli
        self.target_stimulus = [s[l==1][0] for s,l in zip(stimuli,labels)] # Extract the target labels out of it

class TrialIterator(object):
    def __init__(self,data):
        self.data = data
        self.trial_idx = 0

    def __iter__(self):
        return self

    def next(self):
        """
        Iterate over all the stimulus presentations
        :return:
        """

class SupervisedTrialIterator(object):
    def __init__(self,data):
        self.data = data
        self.trial_idx = 0
        self.stimulus_idx = 0

    def __iter__(self):
        return self

    def next(self):
        """
        Iterate over all the stimulus presentations
        :return:
        """
