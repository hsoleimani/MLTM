import numpy as np
from sklearn.svm import SVC
from scipy.sparse import csr_matrix
import re

def miSVM(N, fname, svmC):

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

	bag_labels = np.zeros(len(bags))
	for d,doc in enumerate(sorted(bags.keys())):
		bag_labels[d] = np.max(Y[bags[doc]])


	prev_Y = Y.copy()

	clf = SVC(C=float(svmC), kernel='linear')
	cnt = 0

	while True:
		if len(np.unique(Y))<2:
			break
		clf.fit(X, Y) 

		Y = clf.predict(X)
		ypred_f = clf.decision_function(X)

		change = 0
		bag_pred = np.zeros(len(bags))
		bag_pred_f = np.zeros(len(bags))
		for d,doc in enumerate(sorted(bags.keys())):
			sids = bags[doc]
			bag_pred[d] = np.max(Y[sids])
			bag_pred_f[d] = np.max(ypred_f[sids])
			if (bag_labels[d] == 1) and (bag_pred[d] == -1):
				imax = np.argmax(ypred_f[sids])
				Y[sids[imax]] = 1
				change += 1

		if (np.sum(Y!=prev_Y) == 0) or cnt >= 20:
			break

		prev_Y = Y.copy()		
		cnt += 1

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

	ypred_f = clf.decision_function(X)

	bag_pred_f = np.zeros(len(bags))
	for d,doc in enumerate(sorted(bags.keys())):
		sids = bags[doc]
		bag_pred_f[d] = np.max(ypred_f[sids])


	return(normalize(bag_pred_f), normalize(ypred_f), clf, bags)
