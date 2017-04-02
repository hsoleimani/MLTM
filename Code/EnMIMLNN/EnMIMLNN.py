import numpy as np
from sklearn.svm import SVC
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
import re, os
#import pdb


def read_data(N, fname):

	row = []
	column = []
	data = []
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

		doc = int(ln_split[1].lstrip('d'))
		try:
			bags[doc].append(d)
		except KeyError:
			bags.update({doc:[d]})

		d += 1
		#if d >= 1500:
		#	break

	fp.close()

	X = csr_matrix((np.array(data), (np.array(row), np.array(column))), shape=(len(set(row)),N))

	return X, bags

def clustering(X, bags, K, Tmax, seed0):

	np.random.seed(seed0)

	nbags = len(bags)
	if K > nbags:
		return None, None

	#******* clustering 
	# compute all pair-wise distances
	def Hausdorff_dist(bag1, bag2, X):
		inst_dist = pairwise_distances(X[bag1,:],X[bag2,:])
		
		#return max(np.max(np.min(inst_dist,1)), np.max(np.min(inst_dist,0)))
		h2 = max(np.max(np.min(inst_dist, 1)), np.max(np.min(inst_dist, 0)))
		h3 = np.min(np.min(inst_dist, 1))
		h1 = (np.sum(np.min(inst_dist, 1)) + np.sum(np.min(inst_dist, 0)))/float(inst_dist.shape[0]+inst_dist.shape[1])
		#pdb.set_trace()
		return (h1 + h2 + h3)/3.0

	bag_distances = dict()
	keys = sorted(bags.keys())

	# initial clusters
	KM = KMeans(n_clusters = K, max_iter=50, tol=0.001)
	kmX = np.zeros((len(bags), X.shape[1]))
	for ind, bagid in enumerate(keys):
		kmX[ind, :] = np.asarray(np.mean(X[bags[bagid],:], 0)).reshape(-1)
	kmX = csr_matrix(kmX)
	KM.fit(kmX)
	centers = [keys[x] for x in np.argmin(KM.transform(kmX), 0)]
	'''KM.fit(X)
	bag_labels = np.zeros((len(keys),K))
	for d0,d in enumerate(keys):
		temp = KM.labels_[bags[d]].copy()		
		bag_labels[d0,:] = np.array([np.sum(temp==k) for k in range(K)])
	centers = np.argmax(bag_labels, 0)'''
	#print(centers)

	#centers = np.random.choice(bags.keys(),K,replace=False)
	for t in range(Tmax):
	
		distances = np.zeros((nbags,K))
		for k in range(K):
			#print(k)
			ck = centers[k]
			distances[:,k] = np.array([Hausdorff_dist(bags[ck], bags[d], X) for d in keys])

		assignments = np.argmin(distances,1)

		# update clusters
		change = 0
		for k in range(K):
			members = np.where(assignments==k)[0]
			#print(k,len(members))
			if len(members) == 0:
				continue
			dist = np.zeros(len(members))
			for m,mem in enumerate(members):

				dist[m] = 0
				for d in members:
					pair_name = '%d|%d' %(min(mem,d),max(mem,d))
					try:
						dist[m] += bag_distances[pair_name]
					except KeyError:
						temp = Hausdorff_dist(bags[keys[mem]], bags[keys[d]], X)
						bag_distances.update({pair_name:temp})
						dist[m] += temp

				
			new_center = keys[members[np.argmin(dist)]]
			if new_center != centers[k]:
				change += 1
				centers[k] = new_center#.copy()

		print(t, change)
		if change == 0:
			break

	#******* End of clustering 

	# encode the data

	'''XX = np.zeros((nbags, K))
	for k in range(K):
		ck = centers[k]
		for d0,d in enumerate(keys):
			pair_name = '%d|%d' %(min(ck,d),max(ck,d))
			try:
				XX[:,k] = bag_distances[pair_name]
			except KeyError:
				XX[:,k] = Hausdorff_dist(bags[ck], bags[d], X)
		
    '''

	center_instances = list()
	for k in range(K):
		ind = bags[centers[k]]
		center_instances.append(X[ind,:].copy())

	return center_instances, centers


'''def encode_test_data(X, bags, centers):

	K = len(centers)
	nbags = len(bags)
	distances = np.zeros((nbags,K))
	sent_dist = np.zeros((X.shape[0],K))
	for k,center in enumerate(centers):
		ninst = center.shape[0]
		dist = pairwise_distances(X, center)
		sent_dist[:,k] = np.max(dist, 1)
		for d,doc in enumerate(sorted(bags.keys())):
			ind = bags[doc]
			distances[d,k] = max(np.max(np.min(dist[ind,:],1)),np.min(np.max(dist[ind,:],0)))

	return distances, sent_dist'''
		
def normalize(y, C):
	for c in range(C):
		m1 = np.min(y[:,c])
		M1 = np.max(y[:,c])
		if np.sign(m1)!=np.sign(M1):
			y[y[:,c]>0,c] /= M1
			y[y[:,c]<0,c] /= np.fabs(m1)
			y[:,c] = 0.5*(y[:,c]+1)
		elif m1!=M1:
			y[:,c] = (y[:,c]-m1)/(M1-m1)
	return(y)
