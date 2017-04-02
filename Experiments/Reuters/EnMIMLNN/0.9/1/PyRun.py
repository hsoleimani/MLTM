import numpy as np
import os, re, sys, pickle
#sys.path.insert(0,'/gpfs/home/hus152/anaconda2/lib/python2.7/site-packages/')

import pdb

from sklearn.svm import SVC
from sklearn import metrics
from scipy.sparse import csr_matrix
from multiprocessing.dummy import Pool as ThreadPool 
from sklearn.externals import joblib
from sklearn.metrics import pairwise_distances

sys.path.append('/'.join(os.getcwd().split('/')[:-4]))
import myAUC

sys.path.append('../../../../../Code/EnMIMLNN')
import EnMIMLNN

dirnum = int(os.getcwd().split('/')[-1])-1
prop = os.getcwd().split('/')[-2]

nthreads = int(sys.argv[1])

seed0 = 100001 + 100*dirnum
np.random.seed(seed0)

Tmax = 100

dirpath = 'dir'
os.system('mkdir -p %s' %dirpath)
resfile = 'results.txt' 
fpres = open(resfile, 'w')
fpres.write('roc roc_macro fprAUC\n')
fpres.close

Datapath = '../../../../../Data/%s' %os.getcwd().split('/')[-4]

trfile = '%s/train-data.dat' %Datapath
if (prop != '1'):
        trlblfile = '%s/train-label%s.dat' %(Datapath,prop)
else:
        trlblfile = '%s/train-label.dat' %Datapath
tfile = '%s/test-data.dat' %Datapath
tlblfile = '%s/test-label.dat' %Datapath
vfile = '%s/test-data.dat' %Datapath
vlblfile = '%s/test-label.dat' %Datapath
vocabfile = '%s/vocabs.txt' %Datapath
trslblfile = '%s/train-sentlabel.dat' %Datapath
tslblfile = '%s/test-sentlabel.dat' %Datapath
sent_trfile = '%s/train-data.dat' %Datapath
sent_tfile = '%s/test-data.dat' %Datapath


trlbl = np.loadtxt(trlblfile)
tlbl = np.loadtxt(tlblfile)
(Dtr, C) = trlbl.shape
Dt = tlbl.shape[0]
N = len(open(vocabfile).readlines())

def read_docs(docfile, lbld_docs, lbls, outfile):
	fp = open(docfile)
	C = lbls.shape[1]
	fpout = open(outfile, 'w')
	d = 0
	d0 = 0
	while True:
		ln = fp.readline()
		if len(ln) == 0:
			break
		if d not in lbld_docs:
			d += 1
			continue
		Sd = int(re.findall('^<([0-9]*)>',ln)[0])
		sents = re.findall('<[0-9]*?>([0-9 ]*)',ln)
		for s,sent in enumerate(sents[1:]):
			doc = {}
			words = sent.split()
			Ls = len(words)
			for w in words:
				try:
					doc[w] += 1.0/Ls
				except KeyError:
					doc.update({w:1.0/Ls})
			txt = ','.join(['%d:%f' %(int(w)+1,doc[w]) for w in doc.keys()])
			lbls_txt = '|'.join(['%d' %int(c) for c in lbls[d,:]])
			fpout.write('s_%d_d_%d,d%d,%s,%s\n' %(s,d0,d0,lbls_txt,txt))
		d += 1
		d0 += 1
	fp.close()
	fpout.close()
	

def Hausdorff_dist(X1, X2):
	inst_dist = pairwise_distances(X1, X2)
	
	#return max(np.max(np.min(inst_dist,1)), np.max(np.min(inst_dist,0)))
	h2 = max(np.max(np.min(inst_dist, 1)),np.max(np.min(inst_dist, 0)))
	h3 = np.min(np.min(inst_dist, 1))
	h1 = (np.sum(np.min(inst_dist, 1)) + np.sum(np.min(inst_dist, 0)))/float(inst_dist.shape[0]+inst_dist.shape[1])
	return (h1 + h2 + h3)/3.0

# read training docs
ind = np.where(trlbl[:,0] != -1)[0]
trlbl0 = trlbl.copy()
trlbl = trlbl[ind,:].copy()
read_docs(trfile, ind, trlbl0, 'dir/trfile')

# test docs
ind = np.where(tlbl[:,0] != -1)[0]
read_docs(tfile, ind, tlbl, 'dir/tfile')

# validation files
vlbl = np.loadtxt(vlblfile)
ind = np.where(vlbl[:,0] != -1)[0]
read_docs(vfile, ind, vlbl, 'dir/vfile')

pool = ThreadPool(nthreads) 
mu_list = [0.2, 0.4, 0.6, 0.8, 1]
alpha_list = [0.01, 0.05, 0.1]
if os.path.isfile('dir/vccr.txt'):
    vccr = np.loadtxt('dir/vccr.txt')
else:
    vccr = np.zeros((len(mu_list), len(alpha_list)))


# read data
(X, bags) = EnMIMLNN.read_data(N, 'dir/trfile')

# create indices and bags for Us
U_X_indices = []
U_bags = []
keys = sorted(bags.keys())
for c in range(C):
	
	ind = np.where(trlbl[:, c] == 1)[0] 

	inst_cnt = 0
	bag_cnt = 0
	temp_bag = dict()
	temp_inst = list()
	for bag_id in ind:
	    #temp_bag.update({bag_cnt:[i+inst_cnt for i in range(len(bags[bag_id]))]})
	    temp_bag.update({keys[bag_id]:[i for i in bags[keys[bag_id]]]})
	    #temp_inst.extend(bags[bag_id])
	    #inst_cnt += len(temp_bag[bag_cnt])
	    bag_cnt += 1
	    
	U_bags.append(temp_bag)
	U_X_indices.append(temp_inst)
	

for alpha_ind, alpha in enumerate(alpha_list):

	if np.all(vccr[:, alpha_ind] != 0):
	    continue

	# num clusters
	M = [1+int(np.ceil(alpha * len(u_bag))) if len(u_bag)>1 else 1 for u_bag in U_bags]

	# perform clustering
	seed = np.random.randint(seed0)
	centers = [list() for c in range(C)]
	centers_ids = [list() for c in range(C)]
	def clustering_c(c):

	    global centers
	    #centers[c], centers_ids[c] = EnMIMLNN.clustering(X[U_X_indices[c],:], U_bags[c], M[c], Tmax, seed)
	    centers[c], centers_ids[c] = EnMIMLNN.clustering(X, U_bags[c], M[c], Tmax, seed)

	
	pool.map(clustering_c, [c for c in range(C)])
	#for c in range(C):
	#	clustering_c(c)

	# compute delta
	centers_flattened = list()
	for c in range(C):
	    centers_flattened.extend(centers[c])
	num_clusters = len(centers_flattened)


	temp_sum = 0
	for i1, x1 in enumerate(centers_flattened):
	    temp_sum += np.sum(np.array([Hausdorff_dist(x1, x2) for x2 in centers_flattened[i1+1:]]))

	for mu_ind, mu in enumerate(mu_list):
	
	    if vccr[mu_ind, alpha_ind] != 0:
	        continue

	    delta = mu * temp_sum / (num_clusters*(num_clusters-1)/2.0)

	    # compute phi
	    phi = [list() for c in range(C)]
	    #for c in range(C):
	    def compute_phi(c):
	        global phi
	        H = np.zeros((len(bags), M[c]))
	        for i, bag in enumerate(bags.keys()):
	            for j in range(M[c]):
	        
	                H[i, j] = Hausdorff_dist(X[bags[bag], :], centers[c][j])

	        #pdb.set_trace()
	        phi[c] = np.exp( -0.5 * (H/delta)**2)

	    pool.map(compute_phi, [c for c in range(C)])

	    #pdb.set_trace()
	    ## estimate w
	    from numpy.linalg import lstsq
	    W = [list() for c in range(C)]
	    for c in range(C):
	        W[c] = lstsq(phi[c], 2*trlbl[:, c]-1)[0]


	    ###### prediction on validation set
	    (vX, vbags) = EnMIMLNN.read_data(N, 'dir/vfile')
	    vphi = [list() for c in range(C)]
	    def valid_compute_phi(c):
	        global vphi
	        H = np.zeros((len(vbags), M[c]))
	        for i, bag in enumerate(vbags.keys()):
	            for j in range(M[c]):
	        
	                H[i, j] = Hausdorff_dist(vX[vbags[bag], :], centers[c][j])

	        vphi[c] = np.exp( -0.5 * (H/delta)**2)
	        #pdb.set_trace()
	    pool.map(valid_compute_phi, [c for c in range(C)])

	    valid_pred = np.zeros(vlbl.shape)
	    for c in range(C):
	        valid_pred[:, c] = np.dot(vphi[c], W[c])

	    #pdb.set_trace()

	    (roc, roc_macro) = myAUC.compute_auc(EnMIMLNN.normalize(valid_pred, C), vlbl, npts=100)

	    vccr[mu_ind, alpha_ind] = roc
	    
	    print(alpha, mu, roc)
	    os.system('mkdir -p dir/tmp%d_%d' %(alpha_ind, mu_ind))
	    np.savetxt('dir/vccr', vccr, '%f')
	    pickle.dump(W, open('dir/tmp%d_%d/W' %(alpha_ind, mu_ind), 'wb'))
	    pickle.dump(delta, open('dir/tmp%d_%d/delta' %(alpha_ind, mu_ind), 'wb'))
	    pickle.dump(centers, open('dir/tmp%d_%d/center' %(alpha_ind, mu_ind),'wb'))
		    

# find best hyper-params
ind = np.unravel_index(vccr.argmax(), vccr.shape)
alpha_ind = ind[1]
mu_ind = ind[0]
alpha = alpha_list[alpha_ind]
mu = mu_list[mu_ind]

#load W, centers
centers = pickle.load(open('dir/tmp%d_%d/center' %(alpha_ind, mu_ind), 'rb'))
W = pickle.load(open('dir/tmp%d_%d/W' %(alpha_ind, mu_ind), 'rb'))
delta = pickle.load(open('dir/tmp%d_%d/delta' %(alpha_ind, mu_ind), 'rb'))

###### prediction on test set
(tX, tbags) = EnMIMLNN.read_data(N, 'dir/tfile')
M = [len(cent) for cent in centers]
tphi = [list() for c in range(C)]
def test_compute_phi(c):
    global vphi
    H = np.zeros((len(tbags), M[c]))
    for i, bag in enumerate(tbags.keys()):
        for j in range(M[c]):
    
            H[i, j] = Hausdorff_dist(tX[tbags[bag], :], centers[c][j])

    tphi[c] = np.exp( -0.5 * (H/delta)**2)

pool.map(test_compute_phi, [c for c in range(C)])

test_pred = np.zeros(tlbl.shape)
for c in range(C):
    test_pred[:, c] = np.dot(tphi[c], W[c])


(roc, roc_macro) = myAUC.compute_auc(EnMIMLNN.normalize(test_pred, C), tlbl, npts=100)


# ThFprAUC for documents with no labels
nolbld = np.where(np.sum(tlbl,1)==0)[0]
if len(nolbld) > 0:
	TH = np.linspace(0,1,100)
	fpr = np.zeros(len(TH))
	for t,th in enumerate(TH):
		pred = np.round(test_pred[nolbld] > th)
		tn = np.sum((1-pred) == 1)
		fp = np.sum(pred == 1)
		fpr[t] = fp/float(fp+tn)
	fprAUC = metrics.auc(TH,fpr)
else:
	fprAUC = 0


fpres = open(resfile, 'a')
fpres.write('%f %f %f\n' %(roc, roc_macro, fprAUC))
fpres.close()	

os.system('rm -r dir')

