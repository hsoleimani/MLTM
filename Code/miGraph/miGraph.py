import numpy as np
from sklearn.svm import SVC
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances
import re

class Kernel:
	
	def __init__(self, svmGamma):
		self.X = []
		self.Y = []
		self.svmGamma = svmGamma
		self.status = 1
		self.kern_dict = dict()
		self.saved_dist = None
	
	def read_data(self, N, fnamelist):

		row = []
		column = []
		data = []
		Y = []
		d = 0
		self.bags_list = list()

		for app,fname in zip(['tr','v','t'],fnamelist):
			bags = dict()
			fp = open(fname)
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
				doc = app + ln_split[1].lstrip('d')
				try:
					bags[doc].append(d)
				except KeyError:
					bags.update({doc:[d]})

				d += 1

			fp.close()
			self.bags_list.append(bags)

		self.Y = np.array(Y)
		self.X = csr_matrix((np.array(data), (np.array(row), np.array(column))), shape=(len(set(row)),N))

	# compute weight
	def w_func(self, ind):
		n = len(ind)
		dist = pairwise_distances(self.X[ind,:], self.X[ind,:])
		delta = np.mean(dist)
		w = (dist<delta).astype(np.float) + 1e-5
		return 1.0/np.sum(w,1)

	def compute_weight(self):
		self.weights = {}
		#self.sum_weights = {}
		for bags in self.bags_list:
			for d in bags.keys():
				temp = self.w_func(bags[d]).copy()
				self.weights.update({d:temp/np.sum(temp)})
				#self.sum_weights.update({d:np.sum(self.weights[d])})
	
	def precom_dist(self, i):

		ind2 = []
		trkeys = sorted([int(x.lstrip('tr')) for x in self.bags_list[0]])
		for key in trkeys:
			ind2.extend(self.bags_list[0]['tr%d' %key])
		ind1 = self.bags_list[self.status-1][i]


		self.dist = np.dot(self.weights[i].T,np.exp(-self.svmGamma*pairwise_distances(self.X[ind1,:],self.X[ind2,:])**2))

	'''def comp_kij(self, i, j, ind1, ind2):

		if self.saved_dist != i:
			self.precom_dist(i)
			self.saved_dist = i
		
		return np.dot(self.dist[ind2],self.weights[j])#/(self.sum_weights[i]*self.sum_weights[j])'''
		

	def my_kernel(self, xx, yy):
		K = np.zeros((xx.shape[0], yy.shape[0]))
		app = ['tr','v','t'][self.status-1]
		id0 = self.status-1

		for i,x in enumerate(xx[:,0]):
			ind1 = self.bags_list[id0][app+str(int(x))]
			xx = app+str(int(x))
			if self.saved_dist != xx:
				self.precom_dist(xx)
				self.saved_dist = xx
			K[i,:] = np.array([np.dot(self.dist[self.bags_list[0]['tr%d'%int(y)]],self.weights['tr%d'%int(y)]) for y in yy[:,0]])
			#K[i,:] = np.array([self.comp_kij(app+str(int(x)), 'tr%d'%int(y), ind1, self.bags_list[0]['tr%d'%int(y)]) for y in yy[:,0]])

		return K


def normalize(y):
	m1 = np.min(y)
	M1 = np.max(y)
	if np.sign(m1)!=np.sign(M1):
		y[y>0] /= M1
		y[y<0] /= np.fabs(m1)
		y = 0.5*(y+1)
	elif m1==M1:
		return y
	else:
		y = (y-m1)/(M1-m1)
	return y


