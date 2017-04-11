"""
Batch experiment

The unsupervised class
"""

import numpy as np
import config
from tools.fileio import load
from decoder.erp_decoder import UnsupervisedEM, LDADecoder
from sklearn.metrics import roc_auc_score


subject = 'VPkw'
_n_iterations = 100
_n_commands = 6 # Keep this at 6!

print "Subject: %s" % subject

# Load the data
data_calib, data_test = load('%s/amuse_%s.pkl'%(config._processed,subject))
_n_dim = np.prod(data_test.eeg.shape[2:])
x_train, y_train = data_calib.get_data_as_xy()
x, y = data_test.get_data_as_xy()
truth = data_test.target_stimulus


print('data is loaded...')
lda_decoder = LDADecoder(_n_commands,x_train,y_train)
lda_decoder.add_data(data_test)
lda_auc = roc_auc_score(y, lda_decoder.apply_single_stimulus(x))
lda_acc = 100.0*np.mean(lda_decoder.predict_all_trials()==truth)

em_decoder = UnsupervisedEM(_n_dim,_n_commands)
em_decoder.add_data(data_test)
print ('data is put into the decoder...')

for i in range(_n_iterations):
    em_decoder.update_decoder()
    pred = em_decoder.predict_all_trials()
    print "it: %d\nEM:  symbol acc: %.2f, auc: %.2f,\nLDA: symbol acc: %.2f, auc: %.2f"%(i, 100.0*np.mean(pred==truth),roc_auc_score(y, em_decoder.apply_single_stimulus(x)),lda_acc,lda_auc)

