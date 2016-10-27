import numpy as np
from sklearn.svm import SVC
from scipy.sparse import csr_matrix
import re

def MISVM(N, fname, svmC):

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


	num_neg = 0
	num_pos = 0
	MI_nsamples = 0
	bag_labels = np.zeros(len(bags))
	for d,doc in enumerate(sorted(bags.keys())):
		bag_labels[d] = np.max(Y[bags[doc]])
		if (bag_labels[d] == 1):
			num_pos += 1
			MI_nsamples += 1
		else:
			num_neg += 1
			MI_nsamples += len(bags[doc])


	row = []
	column = []
	data = []
	MI_Y = []
	selector = dict()
	MId = -1
	mapping = dict()
	for d,doc in enumerate(sorted(bags.keys())):
		if (bag_labels[d] == 1):
			temp = X[bags[doc],:].mean(0).copy()
		else:
			temp = X[bags[doc],:].copy()
		nzind = temp.nonzero()
		i0 = 0
		ilist = []
		for i,j in zip(np.array(nzind[0]).reshape(-1,),np.array(nzind[1]).reshape(-1,)):

			if i not in ilist:
				MId += 1
				ilist.append(i)
				i0 = ilist.index(i)
				if (bag_labels[d] == 1):
					MI_Y.append(1)
					selector.update({MId:0})
					mapping.update({doc:MId})
				else:
					MI_Y.append(-1)

			row.append(MId)
			column.append(j)
			data.append(temp[i,j])

	MI_X = csr_matrix((np.array(data), (np.array(row), np.array(column))), shape=(len(set(row)),N))
	MI_Y = np.array(MI_Y)


	clf = SVC(C=float(svmC), kernel='linear')
	cnt = 0

	while True:
		clf.fit(MI_X, MI_Y) 

		#ypred = clf.predict(X)
		ypred_f = clf.decision_function(X)
	
		change = 0
		bag_pred = np.zeros(len(bags))
		bag_pred_f = np.zeros(len(bags))
		for d,doc in enumerate(sorted(bags.keys())):
			sids = bags[doc]
			bag_pred_f[d] = np.max(ypred_f[sids])
			bag_pred[d] = np.sign(bag_pred_f[d])
			if (bag_labels[d] == 1):
				imax = sids[np.argmax(ypred_f[sids])]

				MId = mapping[doc]
				if imax != selector[MId]:
					selector[MId] = imax
					change += 1
					nzind = MI_X[MId,:].nonzero()[1]
					for j in np.array(nzind).reshape(-1,):
						MI_X[MId,j] = 0
					nzind = X[imax,:].nonzero()
					for i,j in zip(np.array(nzind[0]).reshape(-1,),np.array(nzind[1]).reshape(-1,)):
						MI_X[MId,j] = X[imax,j]
				
		print (change, cnt)
		if (change == 0) or (cnt >= 20):
			break

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

	return(normalize(bag_pred_f), normalize(ypred_f), clf, bags)
