# prepare training label files for semi-supervised learning
import numpy as np

trfile = 'DBpedia/train-label.dat'
#trfile = 'Synthetic/lbls.txt'

trlbl = np.loadtxt(trfile, dtype = np.int)
np.random.seed(100000001)
plist = [0.1, 0.3, 0.6, 0.8, 0.9,0.05,0.01]
Dtr = trlbl.shape[0]
for p in plist:
	fname = trfile.split('.dat')[0] + str(p) + '.dat'

	ind = np.random.choice(Dtr, int((1-p)*Dtr), replace = False)
	templbl = trlbl.copy()
	templbl[ind,:] = -1
	class_cnt = np.sum(templbl==1,0)
	no_sample = np.where(class_cnt==0)[0]
	for c in no_sample:
		class_ind = np.where(trlbl[:,c] == 1)[0]
		ind = np.random.choice(class_ind, 1)[0]
		templbl[ind,:] = trlbl[ind,:].copy()
	np.savetxt(fname, templbl, '%d')
