class ERPDecoder(object):
    """
    Class to perform ERP Experiments

    I will use the following naming conventions.

    Trial: this contains all the data to predict execute a single command with the BCI.
    Iteration: An iteration is the part of the trial that contains each stimulus once.
    Stimulus: a single event followed by a target or non-target response

    """

    def new_trial(self):
        """

        :return:
        """
        raise NotImplementedError("Subclass responsability")

    def add_iteration(self,x,s):
        """
        Add the iteration to the last trial
        :param x:
        :param s:
        :return:
        """
        raise NotImplementedError("Subclass responsability")

    def predict_last_trial(self):
        """
        Make a prediction of the last trial

        :return:
        """
        raise NotImplementedError("Subclass responsability")

    def predict_all_trials(self):
        """
        Make a prediction of all trials
        :return:
        """
        raise NotImplementedError("Subclass responsability")


class SupervisedERPDecoder(ERPDecoder):
    def train(self,x,y):
        raise NotImplementedError("Subclass responsability")


class AdaptiveERPDecoder(ERPDecoder):
    def update_decoder(self):
        raise NotImplementedError("Subclass responsability")


class UnsupervisedERPDecoder(AdaptiveERPDecoder):
    def __init__(self):
        pass



class LDAERPDecoder(SupervisedERPDecoder):
    pass
