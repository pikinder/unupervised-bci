import numpy as np
import scipy as sp
from tools.fileio import load
from decoder.legacy import p300_speller_unigram
import config
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
plt.ion()

subject = 'VPfas'

data,data_test = load('%s/amuse_%s.pkl'%(config._processed,subject))
print('data is loaded...')

_data_dim = np.prod(data.eeg.shape[2:])
_nr_commands = 6




'''INITIALIZATION'''
sigma_t = 1.0*np.eye(1)
delta_w =10.0*np.eye(_data_dim)
mu_w = np.zeros((_data_dim,1))
init_w = np.random.randn(_data_dim,1)


'''
    Build classifiers
'''
speller = p300_speller_unigram(w=init_w,mu_w=mu_w,delta_w=delta_w,sigma_t=sigma_t,nr_commands=_nr_commands,max_delta_w=10.**3.,prior_command_log_probs=-np.log(_nr_commands)*np.ones(_nr_commands))

tot_data = []
tot_label = []
# Add data to speller
for x,y,s in zip(data.eeg,data.labels,data.stimuli):
    x = x[:,::,:].reshape(x.shape[0],-1)
    x = x-x.mean(axis=1)[:,np.newaxis]
    x = x/np.std(x,axis=1)[:,np.newaxis]
    tot_data.append(x)
    tot_label.append(y)
    speller.add_letter([x],[s.T])
tot_data = np.vstack(tot_data)
tot_label = np.hstack(tot_label)

# Run for 100 iterations
for k in range(100):
    print "\n\nIteration: %d"%k

    # Update the model
    if k != 0:
        speller._maximization()
    speller._expectation()

    # Check selection accuracy
    selected_stimuli = np.argmax(speller.probs[:,:],axis=1)
    correct_offline=np.sum(data.target_stimulus==selected_stimuli)
    print "true: ", data.target_stimulus
    print "pred: ", selected_stimuli
    print 'correct: %.2f' % (100.0*correct_offline/selected_stimuli.shape[0],)

    # Check auc on individiual stimuli
    stim_probs = speller.do_individual_intens(tot_data)
    auc = roc_auc_score(tot_label,stim_probs)
    print "auc: %.3f" %(auc,)

    # Print statistics
    print ""
    print "sigma_t %.2f delta_w: %.2f" % ( speller.sigma_t[0,0], speller.delta_w[0,0])
