# EEGForceMap
The official repository of the work "Improving Continuous Grasp Force Decoding from EEG with Time-Frequency Regressors and Premotor-Parietal Network Integration" by Parth G. Dangi and Yogesh K. Meena, IEEE SMC (2025). 

# Installing Dependencies
Use the versions of the packages listed in Requirements.txt in Python environment 3.7 or above

# Reproducing results
Dataset - Use the WAY-EEG-GAL Dataset developed by Luciw et.al. for this experiment. The instructions of selecting the EEG files from WAY-EEG-GAL dataset is given in ReadMe file in Code folder
Data - The Data folder consists of two files, The file written 'Each Feature and Force Data' consists of the ERP, PSD and ERDS features used in the original study and their respective force values, while the 'Model Performance Data' shows the performance results of the models developed in the original study. 
Code - This folder consists of three subfolders, these are:
1. EEGForceMap with Regressors: contains all the four regressor models (Simple Linear Regressor, Multiple Linear Regressor, Partial Least Square Regressor and Neural Network Regressor) which were integrated with EEGForceMap.
2. Feature Analysis: Consists of Codes required for reading and plotting the Excel file given in the Data folder in the folder 'Each Feature and Force Data'
3. Post-hoc Analysis: This folder contains codes developed for post-hoc analysis of model's perfomance as well as code for ablation study and developing subject-specific and subject-independent models.

In all these codes, the file path inputs are replaced with a placeholder captions, Before using, fill the designated file path in place of placeholders.
