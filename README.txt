This repository contains the source code and data files used in the experiments reported in the following paper:

H. Soleimani, D. J. Miller, "Semisupervised, Multilabel, Multi-Instance Learning for Structured Data," Neural Computation, vol. 29, no. 4, pp. 1053-1102, 2017.

Please see the paper for details of the algorithms and the experiments. 

This paper is an extension of an earlier work presented in CIKM2016. Please see the branch CIKM2016 for the experiments reported in that paper.

Contents:

1. The "Code" folder contains the source code for MLTM, MLTMVB, PLLDA, SSLDA, LR, MIL (MISVM, mi_SVM), EnMIMLNN, miGraph, EM-DD, and LDA. Compile each program in a Linux-based system by typing "make".

2. The "Data" folder contains the training and test data sets we used in our experiments in the paper. In most cases, it also contains necessary python scripts to download the data, do the required pre-processing, and split the data into training and test sets. Essentially, running "prepare_ohsumed.py" for instance, is enough to generate the same training and test sets used in the paper.

3. The actual experiments are in the "Experiments" folder. For each dataset and every label proportion, we have a folder in the path: "Dataset/Method/prop/rep/" where Dataset = {Ohsumed, DBPedia, Delicious, Reuters}, Method = {LR, MLTM, MLTMVB, PLLDA, SLDA, MISVM, mi_SVM, migraph, EMDD}, prop = {0.01, 0.05, 0.1, 0.3, 0.6, 0.8, 1} (the label proportion in the training set), and rep = {1,2,3,4,5}; for each method and every label proportion, we repeat the experiments with 5 different initialization, and then take average. Due to time constraint, we had to separate them and run them in parallel. 
For some methods, only one trial (rep={1}) was performed. These methods are miSVM, MISVM, and miGraph, which are insensitive to initialization, and EnMIMLNN, whose computation was
found to be prohibitive.

In each folder, there is a python script ("PyRun.py") which takes care of all steps: training, test, and computing ROC curves.

After running all experiments for each Dataset/Method, the python script "semisup_results.py" in the experiments folder takes average over all 5 iterations and save the final results in "Experiments/Results".

Note: The DLM model in the paper is called LR in these files.
