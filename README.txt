This repository contains the source code and data files used in the experiments reported in the following paper:

Soleimani, H. and Miller, D. J. (2016). Semi-supervised multi-label topic models for document classification and sentence labeling. In CIKM, pages 105-114, DOI: 10.1145/2983323.2983752.

Please see the paper for details of the algorithms and the experiments. 

Contents:

1. The "Code" folder contains the source code for MLTM, PLLDA, SSLDA, LR, MISVM, mi_SVM, and LDA. Compile each program in a Linux-based system by typing "make".

2. The "Data" folder contains the training and test data sets we used in our experiments in the paper. In most cases, it also contains necessary python scripts to download the data, do the required pre-processing, and split the data into training and test sets. Essentially, running "prepare_ohsumed.py" for instance, is enough to generate the same training and test sets used in the paper.

3. The actual experiments are in the "Experiments" folder. For each dataset and every label proportion, we have a folder in the path: "Dataset/Method/prop/rep/" where Dataset = {Ohsumed, DBPedia, Delicious}, Method = {LR, MLTM, PLLDA, SLDA, MISVM, mi_SVM}, prop = {0.01, 0.05, 0.1, 0.3, 0.6, 0.8, 0.9, 1} (the label proportion in the training set), and rep = {1,2,3,4,5}; for each method and every label proportion, we repeat the experiments with 5 different initialization, and then take average. Due to time constraint, we had to separate them and run them in parallel. 

In each folder, there is a python script ("PyRun.py") which takes care of all steps: training, test, and computing ROC curves.

After running all experiments for each Dataset/Method, the python script "semisup_results.py" in the experiments folder takes average over all 5 iterations and save the final results in "Experiments/Results".

Note: The DLM model in the paper is called LR in these files.
