"""
Batch experiment

The unsupervised class
"""

import numpy as np
import config
from tools.fileio import load
from decoder.erp_decoder import UnsupervisedEM
from sklearn.metrics import roc_auc_score


subject = 'VPkw'
_n_iterations = 100
_n_commands = 6 # Keep this at 6!

print "Subject: %s" % subject

# Load the data
_,data_test = load('%s/amuse_%s.pkl'%(config._processed,subject))
_n_dim = np.prod(data_test.eeg.shape[2:])
x, y = data_test.get_data_as_xy()

print('data is loaded...')
decoder = UnsupervisedEM(_n_dim,_n_commands)
decoder.add_data(data_test)
print ('data is put into the decoder...')

for i in range(_n_iterations):
    decoder.update_decoder()
    pred = decoder.predict_all_trials()
    truth = data_test.target_stimulus
    print "it: %d, symbol acc: %.2f, auc: %.2f"%(i, 100.0*np.mean(pred==truth),roc_auc_score(y,decoder.apply_single_trial(x)))

