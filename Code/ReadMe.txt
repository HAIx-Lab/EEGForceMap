This is the Code and Data file for the Publication "Improving Continuous Grasp Force Decoding from EEG with Time-Frequency Regressors and Premotor-Parietal Network Integration" By Parth G. Dangi. The file contains these folders.

1. EEGForceMap with Regressors - 

This file contains the codes which integrate the EEGForceMap methodology with the regressor models. Note that each regressor model employs a different version of EEGForceMap, suitable for their input types and requirements. However, each file has these common functions: 

1) Filtering the Data
2) Using ICLabel to perform ICA, label components and remove them
3) Feature Extraction codes (Statistical Features, PSD and ERDS calculation)

Hence, if you modify the codes, Do not mess with these functions. However, you can change the regressor as much as you like.

Also, this study is a pseudo-online experiment, where data is fed in a sliding window approach to ensure online status. The input files to be used are derived from WAY-EEG-GAL Dataset and stored in the file 'Sample for Decoder Model', which contains weight specific runs from WAY-EEG-GAL Dataset (Data Repository -  https://doi.org/10.6084/m9.figshare.988376) Each participant's files from WAY-EEG-GAL Dataset used for this experiment are as follows:
Participant - Run number
1 - 1, 4, 7, 8, 9
2 - 3, 5, 7, 8, 9
3 - 1, 6, 7, 8, 9
4 - 3, 5, 7, 8, 9
5 - 2, 7, 8, 9
6 - 1, 4, 7, 8, 9
7 - 2, 6, 7, 8, 9
8 - 3, 5, 7, 8, 9
9 - 1, 6, 7, 8, 9
10 - 3, 5, 7, 8, 9
11 - 2, 4, 7, 8, 9
12 - 1, 4, 7, 8, 9

2. Post-hoc Analysis - 

This folder contains the codes that plots and performs post-hoc ablation study along with implementing the EEGForceMap in subject-specific and independent conditions. All the codes except for Ablation studies can be run in a normal desktop. Whereas, ablation studies should be run on the GPU environment with the tensorflow's GPU modification. Also, While using the code, Replace the placeholder phrase 'Sample for Decoder Model' with the file path of the file 'Sample for Decoder Model'.

3. Feature Analysis - 

Use this code to plot the feature's correlation with the grasp force. While using this code, use the file path of Excel file  All_Runs_Features_List_1.xlsx as input. 

