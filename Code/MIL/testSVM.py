import numpy as np
from sklearn.svm import SVC
from scipy.sparse import csr_matrix
import re

def testSVM(N, fname, clf):

	row = []
	column = []
	data = []
	Y = []
	bags = dict()
	fp = open(fname)
	d = 0
	while True:
		ln = fp.readline()
		if len(ln)==0:
			break
		ln_split = ln.split(',')
		wrds = re.findall(r',([0-9\.]*):[0-9\.]*', ln)
		cnts = re.findall(r',[0-9\.]*:([0-9\.]*)', ln)
		for i,wrd in enumerate(wrds):
			row.append(d)
			column.append(int(wrd)-1)
			data.append(float(cnts[i]))

		Y.append(2*int(ln_split[2])-1)
		doc = int(ln_split[1].lstrip('d'))
		try:
			bags[doc].append(d)
		except KeyError:
			bags.update({doc:[d]})

		d += 1

	fp.close()

	Y = np.array(Y)
	X = csr_matrix((np.array(data), (np.array(row), np.array(column))), shape=(len(set(row)),N))

	ypred_f = clf.decision_function(X)


	bag_pred = np.zeros(len(bags))
	bag_pred_f = np.zeros(len(bags))
	for d,doc in enumerate(sorted(bags.keys())):
		sids = bags[doc]
		bag_pred_f[d] = np.max(ypred_f[sids])
		bag_pred[d] = np.sign(bag_pred_f[d])

	def normalize(y):
		m1 = np.min(y)
		M1 = np.max(y)
		if np.sign(m1)!=np.sign(M1):
			y[y>0] /= M1
			y[y<0] /= np.fabs(m1)
			y = 0.5*(y+1)
		else:
			y = (y-m1)/(M1-m1)
		return(y)


	return(normalize(bag_pred_f), normalize(ypred_f), bags)
