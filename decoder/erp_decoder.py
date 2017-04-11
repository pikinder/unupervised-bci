from data.erp_data import ERPData,TrialIterator
import numpy as np

class ERPDecoder(object):
    """
    Class to perform ERP Experiments

    I will use the following naming conventions.

    Trial: this contains all the data to predict execute a single command with the BCI.
    Iteration: An iteration is the part of the trial that contains each stimulus once.
    Stimulus: a single event followed by a target or non-target response

    """
    def __init__(self,n_dim,n_stimuli):
        self.n_dim = n_dim
        self.n_stimuli = n_stimuli

    def add_data(self,data):
        """

        :param data:
        :return:
        """
        assert isinstance(data,ERPData)
        for x,s in TrialIterator(data):
            self.add_trial(x,s)

    def predict_last_trial(self):
        """
        Make a prediction of the last trial

        :return:
        """
        # This code is horribly inefficient but works always
        return self.predict_all_trials()[-1]

    def add_trial(self,x,s):
        """

        :return:
        """
        raise NotImplementedError("Subclass responsability")

    def predict_all_trials(self):
        """
        Make a prediction of all trials
        :return:
        """
        raise NotImplementedError("Subclass responsability")

    def apply_single_trial(self,x):
        """

        :param x:
        :return:
        """
        raise NotImplementedError("Subclass responsability")

class AdaptiveERPDecoder(ERPDecoder):
    """
    An adaptive decoder needs a method to learn from the observed data...
    """
    def update_decoder(self):
        raise NotImplementedError("Subclass responsability")


class UnsupervisedEM(AdaptiveERPDecoder):
    """

    """
    def __init__(self,n_dim,n_stimuli):
        super(UnsupervisedEM,self).__init__(n_dim,n_stimuli)
        from legacy import p300_speller_unigram

        # Variance etc ...
        sigma_t = 1.0*np.eye(1)
        delta_w =10.0*np.eye(self.n_dim+1)
        delta_w[-1,-1]=0 # Do not regularise the bias
        mu_w = np.zeros((self.n_dim+1,1))
        init_w = np.random.randn(self.n_dim+1,1)
        self.speller = p300_speller_unigram(w=init_w,mu_w=mu_w,delta_w=delta_w,sigma_t=sigma_t,nr_commands=self.n_stimuli,max_delta_w=10.**3.,prior_command_log_probs=-np.log(self.n_stimuli)*np.ones(self.n_stimuli))

    def add_trial(self,x,s):
        """

        :return:
        """
        from tools.preprocess import flatten_normalise_bias
        self.speller.add_letter([flatten_normalise_bias(x)],[s.T])

    def predict_all_trials(self):
        self.speller._expectation()
        return self.speller.probs.argmax(axis=1)


    def update_decoder(self):
        self.speller._expectation()
        self.speller._maximization()

    def apply_single_trial(self,x):
        from tools.preprocess import flatten_normalise_bias
        return self.speller.do_individual_intens(flatten_normalise_bias(x))