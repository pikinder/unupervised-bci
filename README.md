# unupervised-bci
This repository contains code implementing an unsupervised decoder for Event-Related Potential based Brain-Computer Interfaces. The following methods are included:
 1. Unsupervised EM [[1]](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0033758).
 2. (Work in progress) Learning from Label Proportions based decoding.
 3. (Work in progress) A supervised baseline using shrinkage LDA.
 
## Usage
 1. Download and preprocess the data by running __setup.sh__
 2. run an _experiment*.py_ script, the different scripts are documented below.

## Experiments
### experiment_amuse_batch.py
This experiment loads the online data from a single subject. It gives the unsupervised classifier access to all data (without labels) and performs several update iterations, in each iteration the selection accuracy and single trial accuracy are printed. This script does not always converge to a good solution and a restart might be required. Tricks to address this issue are discussed in [1]. 

## Datasets
The repository contains code to download (and if needed pre-process) the following datasets. 
 * AMUSE dataset: Auditory 6 class ERP based BCI. The dataset belongs to Schreuder et al [3]. This dataset can only be used with an EM decoder.
 * (work in progress) Learning from Label Proportions BCI. This is the data from our LLP-BCI experiments belonging to Hübner et al [4]. It can be used with an LLP decoder and an EM decoder.
 Please cite the respective papers when these datasets are used.
 
 
## Code was tested using:
 * python 2.7.12
 * sklearn 0.18.1
 * numpy 1.12.1
 * scipy 0.19.0
 
## References
 1. Kindermans et al. [_A bayesian model for exploiting application constraints to enable unsupervised training of a P300-based BCI_](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0033758)
 2. Kindermans et al. [_A P300 BCI for the masses: prior information enables instant unsupervised spelling_](http://papers.nips.cc/paper/4775-a-p300-bci-for-the-masses-prior-information-enables-instant-unsupervised-spelling.pdf)
 3. Schreuder et al. [_Listen, you are writing! Speeding up online spelling with a dynamic auditory BCI_](http://journal.frontiersin.org/article/10.3389/fnins.2011.00112/full)
 4. Hübner et al. [_Learning from Label Proportions in Brain-Computer Interfaces: Online Unsupervised Learning with Guarantees_](https://arxiv.org/abs/1701.07213)
