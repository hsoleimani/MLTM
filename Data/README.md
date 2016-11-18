# DeliciousMIL: A Data Set for Multi-Label Multi-Instance Learning with Instance Labels

### Abstract:
This dataset includes 1) 12234 documents (8251 training, 3983 test) extracted from DeliciousT140 dataset, 2) class labels for all documents, 3) labels for a subset of sentences of the test documents.

### Source:
Hossein Soleimani,  
School of Electrical Engineering and Computer Science, Pennsylvania State University, PA, USA  
hsoleimani@psu.edu  
https://hsoleimani.github.io

David J. Miller,  
School of Electrical Engineering and Computer Science, Pennsylvania State University, PA, USA,  
djmiller@engr.psu.edu

Creators of DeliciousT140 dataset:

Arkaitz Zubiaga, Alberto P. Garc&iacute;a-Plaza, V&iacute;ctor Fresno, and Raquel Mart&iacute;nez,  
Departamento de Lenguajes y Sistemas Inform&aacute;ticos, Universidad Nacional de Educaci&oacute;n a Distancia, Madrid, Spain

### Relevant Information:
This dataset provides ground-truth class labels to evaluate performance of multi-instance learning models on both instance-level and bag-level label predictions. DeliciousMIL was first used in [1] to evaluate performance of MLTM, a multi-label multi-instance learning method, for document classification and sentence labeling.

Multi-instance learning is a special class of weakly supervised machine learning methods where the learner receives a collection of labeled bags each containing multiple instances. A bag is set to have a particular class label if and only if at least one of its instances has that class label.

DeliciousMIL consists of a subset of tagged web pages from the social bookmarking site delicious.com. The original web pages were obtained from DeliciousT140 dataset, which was collected by [2] from the delicious.com in June 2008. Users of the website delicious.com bookmarked each page with word tags. From this dataset, we extracted text parts of each web page and chose 20 common tags as class labels. These class labels are:
reference, design, programming, internet, computer, web, java, writing, English, grammar, style, language, books, education, philosophy, politics, religion, science, history, and culture.

We randomly selected 12234 pages and randomly divided them into 8251 training and 3983 test documents. We also applied Porter stemming and standard stopword removal. 

Each text document is a bag within a multi-instance learning framework consisting of multiple sentences (instances). The goal is to predict document-level and sentence-level class labels on the test set using a model which is trained given only the document-level class labels in the training set.
To evaluate performance of such a model, we have manually labeled 1468 randomly selected sentences from the test documents. Please see [1] for more information. 

### Attribute Information:

1. train-data.dat and test-data.dat:
These files contain the bag-of-word representation of the training and test documents. Each line is of the form:  
```<S_d> sentence_1 sentence_2 … sentence_{Sd}```  
where Sd is the number of sentences in document d. Each sentence s is in the following format:  
```<L_s> w_{1s} w_{2s} … w_{L_s s}```  
where L_s is the number of words in sentence s, and w_{is} is an integer which indexes the i-th term in sentence s. 

2. vocabs.txt: This file contains the list of words used for indexing the document representations in data files. Each line contains: word, index.
3. train-label.dat and test-label.dat:
Each file contains a D by C binary matrix where D is the number of documents in every file and C=20 is the number of classes. The element b_{dc} is 1 if class c is present in document d and zero otherwise.
4. test-sentlabel.dat, labeled_test_sentences.dat:

  * test-sentlabel.dat: This file contains class labels for sentences of the test documents. Each line d is of the form:  
    ```<y_{11d} y_{12d} … y_{1Cd}><y_{21d} y_{22d} … y_{2Cd}>...<y_{S_d1d} y_{S_d2d} … y_{S_dCd}>```  
where y_{scd} is the binary indicator of class c for sentence s of document d. y_{scd} is 1 if class c present in sentence s and zero otherwise.   
Note that only 1468 sentences are randomly selected and manually labeled. For the rest of the sentences that are unlabeled, we set y_{scd}=-1. 
  * labeled_test_sentences.dat: This file only contains the class labels for the 1468 sentences which are manually labeled. Each line of this file is of the form:  
```d s y_{s1d} y_{s2d} … y_{sCd}```  
where d and s are respectively document and sentence indices. 
5. labels.txt: This contains the list of all class labels in this dataset. Each line is of the form: label, index.

Please see https://github.com/hsoleimani/MLTM for example python code for reading these files.

### Relevant Papers:

[1] Hossein Soleimani and David J. Miller. 2016. Semi-supervised Multi-Label Topic Models for Document Classification and Sentence Labeling. In Proceedings of the 25th ACM International on Conference on Information and Knowledge Management (CIKM '16). ACM, New York, NY, USA, 105-114. DOI: http://dx.doi.org/10.1145/2983323.2983752.

[2] Arkaitz Zubiaga, Alberto P. García-Plaza, Víctor Fresno, and Raquel Martínez. 2009. Content-Based Clustering for Tag Cloud Visualization. In Proceedings of the 2009 International Conference on Advances in Social Network Analysis and Mining (ASONAM '09). IEEE Computer Society, Washington, DC, USA, 316-319. DOI=http://dx.doi.org/10.1109/ASONAM.2009.19

### Citation Request:
If you use DeliciousMIL, please cite:

Hossein Soleimani and David J. Miller. 2016. Semi-supervised Multi-Label Topic Models for Document Classification and Sentence Labeling. In Proceedings of the 25th ACM International on Conference on Information and Knowledge Management (CIKM '16). ACM, New York, NY, USA, 105-114. DOI: http://dx.doi.org/10.1145/2983323.2983752.
