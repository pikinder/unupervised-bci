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

subject = 'VPfaz'

data,data_test = load('%s/%s.pkl'%(config._processed,subject))
print('data is loaded...')
def get_xy(data):
    tot_data = []
    tot_label = []
    # Add data to speller
    for x,y,s in zip(data.eeg,data.labels,data.stimuli):
        x = x[:,::].reshape(x.shape[0],-1)
        x = x-x.mean(axis=1)[:,np.newaxis]
        x = x/np.std(x,axis=1)[:,np.newaxis]
        tot_data.append(x)
        tot_label.append(y)
    return np.vstack(tot_data),np.hstack(tot_label)

xt,yt = get_xy(data)
xe,ye = get_xy(data_test)
clf = LDA(solver='eigen',shrinkage='auto')
clf.fit(xt,yt*2-1)
print roc_auc_score(ye*2-1,clf.decision_function(xe))

_data_dim = np.prod(data.eeg.shape[2:])
_nr_commands = 6




'''INITIALIZATION'''
sigma_t = 1.0*np.eye(1)
delta_w =10.0*np.eye(_data_dim)
mu_w = np.zeros((_data_dim,1))
init_w = np.random.randn(_data_dim,1)#np.zeros((data_dim,1))


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
    stimuli = np.zeros((1,x.shape[0]))
    for idx,ss in enumerate(s):
        stimuli[0,idx]=ss
    speller.add_letter([x],[stimuli])
tot_data = np.vstack(tot_data)
tot_label = np.hstack(tot_label)
'''
    Execute EM and evaluate
'''
for k in range(100):
    ''' DO MAXIMIZATION AT THE END, TO GET GOOD EVALUATION OF THE INITIALIZATION '''
    speller._expectation()
    speller_output=speller.do_individual_intens(tot_data)
    print ""
    print "sigma_t %.2f delta_w: %.2f" % ( speller.sigma_t[0,0], speller.delta_w[0,0])
    auc = roc_auc_score(tot_label,speller_output)
    print "auc: %.3f" %(auc,)
    selected_stimuli = np.argmax(speller.probs[:,:],axis=1)
    print "desired: ", data.target_stimulus
    print "selected:", selected_stimuli+1
    correct_offline=np.sum(data.target_stimulus==selected_stimuli)
    print 'correct: %.2f' % (100.0*correct_offline/selected_stimuli.shape[0],)
    if True and auc < 0.5:
        speller.w = -speller.w
        print "AUC HACK"

    ''' DO MAXIMIZATION AT THE END, TO GET GOOD EVALUATION OF THE INITIALIZATION '''
    speller._maximization()
