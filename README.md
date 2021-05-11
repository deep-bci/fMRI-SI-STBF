# fMRI-SI-STBF
Code for fMRI-Informed Bayesian Electromagnetic Spatio-Temporal Extended Source Imaging
'SBFConstruct' generates the spatial basis functions based on EEG-informed cortical clusters using data-driven parcellization, and fMRI-informed clusters;
'Simulation_Data_Generate_fmri_prior' generates simulated EEG data;
'SISTBF' is the source code of SI-STBF, which reconstructs cortical activities based on source matrix decomposation under the empirical Bayesian framework using variational Bayeisan inference. The spatial prior covariance of cortical sources is set as a mixture of EEG-informed and fMRI-informed covariance components (CCs). Using an ARD prior, fMRI-SI-STBF can select the fMRI-informed and EEG-informed CCs that are related to brain activities in an automatic fashion.
