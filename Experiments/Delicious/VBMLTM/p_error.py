import numpy as np
import os, re, sys

Datapath = '../../../Data/Delicious' 
prop = '1'
if (prop != '1'):
	trlblfile = '%s/train-label%s.dat' %(Datapath,prop)
else:
	trlblfile = '%s/train-label.dat' %Datapath

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

			cnt += 1.0
		prob_lbling_error /= cnt 
		prob1 /= cnt 
		prob0 /= cnt 

		fp = open('%s/%s/prob_lbling_error.txt' %(prop, str(random_iter)), 'w')
		fp.write('%f %f %f' %(np.mean(prob_lbling_error), np.mean(prob1), np.mean(prob0)))
		fp.close()

