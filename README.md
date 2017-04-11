# unupervised-bci
This repository contains code implementing an unsupervised decoder for Event-Related Potential based Brain-Computer Interfaces. 

## Decoders implemented
 1. The unsupervised EM based decoder I presented in [_A bayesian model for exploiting application constraints to enable unsupervised training of a P300-based BCI_](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0033758).
 This decoder can be used with the AMUSE dataset
 2. Transfer learning based decoding from our NIPS publication [_A P300 BCI for the masses: prior information enables instant unsupervised spelling_](http://papers.nips.cc/paper/4775-a-p300-bci-for-the-masses-prior-information-enables-instant-unsupervised-spelling.pdf) 
 3. (Work in Progress) Learning from Label Proportions based decoding.
 4. (Work in Progress) A supervised baseline
 
## Datasets
## Usage
 1. Download and preprocess the data by running __setup.sh__
 2. run an _experiment*.py_ script

## Datasets used
 1. The data from the AMUSE paradigm by Schreuder et al.
 
## Implemented experiments
### Amuse Batch

## Code was tested using:
 * python 2.7.12
 * sklearn 0.18.1
 * numpy 1.12.1
 * scipy 0.19.0
 
## Work in progress
 * Inclusion of a script to simulate an online experiment
 * Inclusion of a transfer learning script
 * Inclusion of a script to compare to supervised evaluation
 * Inclusion of Learning from Label Proportions
 
## References
[1] Kindermans et al.
[2] Kindermans et al.
[3] Schreuder et al.
[4] Hübner et al.
