import numpy as np
import scipy as sp
from scipy.misc import logsumexp


class B:
    def mv_normal_cov_like(self, labels, data, sigma_t):
        kwad_deel = np.array(labels - data).flatten()
        exp_deel = np.sum(-1. / 2. / sigma_t[0, 0] * (kwad_deel) ** 2.)
        return np.log(1. / (np.sqrt(np.pi * 2. * sigma_t[0, 0]))) * 1.0 * kwad_deel.shape[0] + exp_deel


compute_gaussian = B()


class p300_speller_base:
    def __init__(self, w, mu_w, delta_w, sigma_t, nr_commands, max_delta_w):
        '''
            Create a new basic P300 speller

            w           -- the initial value for the classifier weight vector
            mu_w        -- the mean of the prior on w
            delta_w     -- If this value is updated automatically, then it is assumed to be isotropic
            sigma_t     -- the variance on the projection into one dimension
            nr_commands -- THe amount of different options to choose from in the speller
            max_delta_w -- The maximum value for the precision on the weight vector. Introduced to avoid singularities (10^3 recommended)
        '''
        self.w = w.copy()
        self.mu_w = mu_w.copy()
        self.delta_w = delta_w.copy()
        self.sigma_t = sigma_t.copy()  # Beta mag geen NP array zijn?

        # Prepare for the data and the stimuli, see description of structure above!
        self.data = []
        self.stimuli = []

        # Meta information which is essential for good P300 working
        self.nr_commands = nr_commands  # How many different characters/commands are there to choose from
        self.max_delta_w = max_delta_w
        self.data_dim = self.w.shape[0]

        self.label = np.array([1., -1.])

        # Caching xTx and xTy for speed reasons
        self.xTx = np.zeros((self.data_dim, self.data_dim))
        self.xTy = [[] for a_i in range(self.nr_commands)]

    def add_letter(self, letter_data, letter_stimuli):
        '''
            add the data for a single letter
            data        --  A numpy array containing the EEG data for character
                            Each row contains the EEG for a single stimulus

            stimuli     --  A numpy array containing the stimuli for this letter. One intensification per column,
                            The rows contain the characters intensified. -1 indicates no valid intensification
                            This value can be used when the number of characters intensified is variable

        '''
        self.data.extend(letter_data)
        self.stimuli.extend(letter_stimuli)
        # update the xtx and xty
        for c_i in range(len(letter_data)):
            self.xTx += np.dot(letter_data[c_i].T, letter_data[c_i])

        # Update cached xTy
        for c_i in range(len(letter_data)):
            # Loop over assigned commands
            for a_i in range(self.nr_commands):
                stimuli_counts = np.sum(letter_stimuli[c_i] == a_i, axis=0)
                data_p = letter_data[c_i][stimuli_counts > 0, :]
                data_n = letter_data[c_i][stimuli_counts == 0, :]

                temp_xTy = np.dot(data_p.T, self.label[0] * np.ones((data_p.shape[0], 1)))
                temp_xTy += np.dot(data_n.T, self.label[1] * np.ones((data_n.shape[0], 1)))
                self.xTy[a_i].append(temp_xTy)

    def add_data_to_letter(self, letter_data, letter_stimuli, letter_index):
        '''
        '''
        self.data[letter_index] = np.vstack((self.data[letter_index], letter_data))
        self.stimuli[letter_index] = np.hstack((self.stimuli[letter_index], letter_stimuli))

        self.xTx += np.dot(letter_data.T, letter_data)
        for a_i in range(self.nr_commands):
            stimuli_counts = np.sum(letter_stimuli == a_i, axis=0)
            data_p = letter_data[stimuli_counts > 0, :]
            data_n = letter_data[stimuli_counts == 0, :]
            self.xTy[a_i][letter_index] += np.dot(data_p.T, self.label[0] * np.ones((data_p.shape[0], 1)))
            self.xTy[a_i][letter_index] += np.dot(data_n.T, self.label[1] * np.ones((data_n.shape[0], 1)))

    def _expectation(self):
        '''
            Execute the entire expectation step
            Stores the marginal probability for each character in self.probs
        '''
        projection = self._compute_projection(self.data)
        likelihoods = self._compute_individual_likelihoods(projection)
        (self.probs, self.data_log_likelihood) = self._compute_character_probabilities(likelihoods, self.stimuli)

    def _maximization(self):
        '''
            Execute all of the maximizations steps
        '''
        self._maximization_w()
        self._maximization_sigma_t()
        self._maximization_delta_w()

    def _compute_character_probabilities(self, likelihoods, stimuli):
        '''

        '''
        print " NOT IMPLEMENTED IN BASE CLASS "

    def _compute_projection(self, data):
        '''
            For all the data, compute the projection into one dimension
            returns a python list with one array per character
        '''
        return [np.dot(data[k], self.w) for k in range(len(data))]

    def _compute_individual_likelihoods(self, projection):
        '''
            Compute the likelihood for the individual projected points
            Returns a list of numpy matrices
            Each element in the list corresponds to a character
            Each row corresponds to a data points
            The columns are first the likelihood for P300, second the non P300 likelihood
        '''
        likelihoods = []
        # Loop over characters
        # print "BUG^^"
        for c_i in range(len(projection)):
            # Build a 2D array which contains the mean squared error between projection and the labels
            # Continue with making it the log of a gaussian :)
            cur_lik = np.tile(np.atleast_2d(projection[c_i]), (1, len(self.label))) - np.atleast_2d(self.label)
            cur_lik = cur_lik ** 2.
            cur_lik *= -0.5 / self.sigma_t[0, 0]
            cur_lik -= np.log(np.sqrt(2.0 * np.pi * self.sigma_t[0, 0]))
            likelihoods.append(cur_lik)
        return likelihoods

    def _maximization_w(self):
        '''
            Maximize the weight vector
        '''
        xTy = np.zeros((self.data_dim, 1))

        # Loop over characters
        for c_i in range(len(self.data)):
            cur_probs = self.probs[c_i, :]

            # Loop over assignations of characters
            for a_i in range(self.nr_commands):
                xTy += cur_probs[a_i] * self.xTy[a_i][c_i]

        # Add the prior
        xTy += np.dot(self.delta_w * self.sigma_t, self.mu_w)

        # Invert it all :), compute w
        self.w = np.dot(np.linalg.inv(self.xTx + self.sigma_t[0, 0] * self.delta_w), xTy)

    def _maximization_delta_w(self):
        '''
            Execute the maximization operation for the precision on the weigth vector
        '''
        self.delta_w = 1. * (self.data_dim) / np.dot(self.w.T - self.mu_w.T, self.w - self.mu_w) * np.eye(self.data_dim)
        if self.delta_w[0][0] >= self.max_delta_w:
            self.delta_w = self.max_delta_w * np.eye(self.delta_w.shape[0])
            print "LIMIT DELTA_W"

    def _maximization_sigma_t(self, projection=None):
        '''
            Maximization update for the precision parameter
        '''
        if projection == None:
            projection = self._compute_projection(self.data)

        self.sigma_t = 0. * self.sigma_t
        number_data_points = sum([projection[n].shape[0] for n in range(len(projection))])
        for c_i in range(len(self.data)):
            cur_probs = self.probs[c_i, :]
            # Loop over assignations
            for a_i in range(self.nr_commands):
                # print "stimulus a_i: ", a_i
                stimuli_counts = np.sum(self.stimuli[c_i] == a_i, axis=0)
                data_p = projection[c_i][stimuli_counts.T > 0, :]
                data_n = projection[c_i][stimuli_counts.T == 0, :]
                # print data_p.shape, "data_n.shape: ", data_n.shape
                # Substract the labels
                data_p -= self.label[0] * np.ones((data_p.shape[0], 1))
                data_n -= self.label[1] * np.ones((data_n.shape[0], 1))
                data_it = np.vstack([data_p, data_n])

                # Add weightd version of projection error to the
                self.sigma_t += cur_probs[a_i] * np.atleast_2d(
                    np.sum(data_it * data_it, axis=0)) / number_data_points  # element_wise mult

    def do_individual_intens(self, tot_data):
        outputs = np.zeros(tot_data.shape[0])
        for i in range(tot_data.shape[0]):
            prob_p = compute_gaussian.mv_normal_cov_like(self.label[0] * np.ones(1),
                                                         np.dot(tot_data[i:i + 1, :], self.w), self.sigma_t)
            prob_n = compute_gaussian.mv_normal_cov_like(self.label[1] * np.ones(1),
                                                         np.dot(tot_data[i:i + 1, :], self.w), self.sigma_t)
            outputs[i] = np.exp(prob_p - logsumexp(np.array([prob_n, prob_p])))
        return outputs


import numpy as np


class online_speller:
    def __init__(self, w, speller_class, *args):
        self.spellers = [speller_class(w, *args), speller_class(-w, *args)]

    def _expectation(self):
        for speller in self.spellers:
            speller._expectation()
        ##
        liks = np.array([speller.data_log_likelihood for speller in self.spellers])
        best_id = np.argmax(liks)
        self.probs = self.spellers[best_id].probs
        self.data_log_likelihood = self.spellers[best_id].data_log_likelihood

    def _maximization(self):
        for speller in self.spellers:
            speller._maximization()

    def select_best_vector_redo_expectation(self):
        liks = np.array([speller.data_log_likelihood for speller in self.spellers])
        best_id = np.argmax(liks)
        worst_id = (best_id + 1) % 2
        self.spellers[worst_id].w = -1.0 * self.spellers[best_id].w.copy()
        self.spellers[worst_id].sigma_t = 1.0 * self.spellers[best_id].sigma_t.copy()
        self.spellers[worst_id].delta_w = 1.0 * self.spellers[best_id].delta_w.copy()
        self.spellers[worst_id]._expectation()

    def add_letter(self, letter_data, letter_stimuli):
        for speller in self.spellers:
            speller.add_letter(letter_data, letter_stimuli)

    def best_probs(self):
        liks = np.array([speller.data_log_likelihood for speller in self.spellers])
        best_id = np.argmax(liks)
        return self.spellers[best_id].probs


class double_init_speller:
    def __init__(self, w, speller_class, *args):
        self.spellers = [speller_class(w, *args), speller_class(-w, *args)]
        self.data = self.spellers[0].data

    def _expectation(self):
        for speller in self.spellers:
            speller._expectation()
        ##
        liks = np.array([speller.data_log_likelihood for speller in self.spellers])
        best_id = np.argmax(liks)
        self.probs = self.spellers[best_id].probs
        self.data_log_likelihood = self.spellers[best_id].data_log_likelihood

    def _maximization(self):
        for speller in self.spellers:
            speller._maximization()

    def add_letter(self, letter_data, letter_stimuli):
        for speller in self.spellers:
            speller.add_letter(letter_data, letter_stimuli)

    def do_individual_intens(self, tot_data):
        liks = np.array([speller.data_log_likelihood for speller in self.spellers])
        best_id = np.argmax(liks)
        return self.spellers[best_id].do_individual_intens(tot_data)
