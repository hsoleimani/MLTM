import numpy as np
import os, re, sys
#from sklearn import metrics
#from sklearn.linear_model import LogisticRegression
#sys.path.insert(0, '/'.join(os.getcwd().split('/')[:-4]))
#import myAUC



#Code = '../../../../../Code/MLTMVB/MLTMVB'
#LDACode = '../../../../../Code/LDA_VB_Parallel/lda_vb'
Datapath = '../../../Data/DBpedia' 
#trfile = '%s/train-data.dat' %Datapath
prop = '1'
if (prop != '1'):
	trlblfile = '%s/train-label%s.dat' %(Datapath,prop)
else:
	trlblfile = '%s/train-label.dat' %Datapath
#tfile = '%s/test-data.dat' %Datapath
#tlblfile = '%s/test-label.dat' %Datapath
#vfile = '%s/valid-data.dat' %Datapath
#vlblfile = '%s/valid-label.dat' %Datapath
#vocabfile = '%s/vocabs.txt' %Datapath
#tslblfile = '%s/test-sentlabel.dat' %Datapath
#ofile = '%s/obsvd-data.dat' %Datapath
#hfile = '%s/lda_hldout-data.dat' %Datapath


for prop in ['0.01', '0.05', '0.1', '0.3', '0.6', '0.8', '0.9', '1']:
	for random_iter in range(1,6):
		print (prop, random_iter)

		dirpath = '%s/%s/dir' %(prop, str(random_iter))


		trlbl = np.loadtxt(trlblfile)
		(Dtr, C) = trlbl.shape

		# compute probability of labeling error
		prob_lbling_error = 0.0
		cnt = 0.0
		slbl_pred = open('%s/001.y' %dirpath).readlines()
		prob_lbling_error = np.zeros(C)
		prob1 = np.zeros(C)
		prob0 = np.zeros(C)
		for d, doc in enumerate(slbl_pred):
			sents_pred = doc.split('|')[:-1]
			pys = np.zeros((len(sents_pred), C))
			for s, sent in enumerate(sents_pred):
				pys[s, :] = np.array([float(x) for x in sent.split()])
			A = np.max(pys, 0)
			B = 1.0 - np.prod(1-pys, 0)

			prob1 += (A * (1.0 - B))
			prob0 += ((1.0 - A) * B)
			prob_lbling_error += ((1.0 - A) * B) + (A * (1.0 - B))

			'''# class 0
			ind = trlbl[d,:]==0
			prob_lbling_error[ind] += ((1.0 - A) * B)[ind]
			cnt_per_class[ind] += 1
			# class -1
			ind = trlbl[d,:]==-1
			prob_lbling_error[ind] += ((1.0 - A) * B)[ind] + (A * (1.0 - B))[ind]
			cnt_per_class[ind] += 1'''

			cnt += 1.0
		prob_lbling_error /= cnt 
		prob1 /= cnt 
		prob0 /= cnt 

		fp = open('%s/%s/prob_lbling_error.txt' %(prop, str(random_iter)), 'w')
		fp.write('%f %f %f' %(np.mean(prob_lbling_error), np.mean(prob1), np.mean(prob0)))
		fp.close()

