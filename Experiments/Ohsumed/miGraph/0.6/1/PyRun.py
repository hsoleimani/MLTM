import numpy as np
import os, re, sys, pickle
#sys.path.insert(0,'/gpfs/home/hus152/.local/lib/python3.3/site-packages/')
sys.path.insert(0,'/gpfs/home/hus152/anaconda2/lib/python2.7/site-packages/')
from sklearn.svm import SVC
from sklearn import metrics
from multiprocessing.dummy import Pool as ThreadPool 
from sklearn.externals import joblib

sys.path.append('/'.join(os.getcwd().split('/')[:-4]))
import myAUC

sys.path.append('../../../../../Code/miGraph')
import miGraph

dirnum = int(os.getcwd().split('/')[-1])-1
prop = os.getcwd().split('/')[-2]

nthreads = int(sys.argv[1])

seed0 = 100001 + 100*dirnum
np.random.seed(seed0)

Tmax = 10

dirpath = 'dir'
os.system('mkdir -p %s' %dirpath)
resfile = 'results.txt' 
fpres = open(resfile, 'w')
fpres.write('roc roc_macro fprAUC\n')
fpres.close

Datapath = '../../../../../Data/Ohsumed'

trfile = '%s/train-data.dat' %Datapath
if (prop != '1'):
        trlblfile = '%s/train-label%s.dat' %(Datapath,prop)
else:
        trlblfile = '%s/train-label.dat' %Datapath
tfile = '%s/test-data.dat' %Datapath
tlblfile = '%s/test-label.dat' %Datapath
vfile = '%s/valid-data.dat' %Datapath
vlblfile = '%s/valid-label.dat' %Datapath

vocabfile = '%s/vocabs.txt' %Datapath
#trslblfile = '%s/train-sentlabel.dat' %Datapath
#tslblfile = '%s/test-sentlabel.dat' %Datapath
#sent_trfile = '%s/train-data.dat' %Datapath
#sent_tfile = '%s/test-data.dat' %Datapath


trlbl = np.loadtxt(trlblfile)
tlbl = np.loadtxt(tlblfile)
(Dtr, C) = trlbl.shape
Dt = tlbl.shape[0]
N = len(open(vocabfile).readlines())

def read_docs(docfile, lbld_docs, lbls, outfile):
	fp = open(docfile)
	C = lbls.shape[1]
	fpout = list()
	for c in range(C):
		fpout.append(open(outfile+'_'+str(c), 'w'))
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
			for c in range(C):
				fpout[c].write('s_%d_d_%d,d%d,%d,%s\n' %(s,d0,d0,lbls[d,c],txt))
		d += 1
		d0 += 1
	fp.close()
	for c in range(C):
		fpout[c].close()
	

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


def train_model_class(c):

	global svmC
	global svmGamma
	global c0
	global valid_pred

	fname_list = ['dir/trfile_%d' %c, 'dir/vfile_%d' %c, 'dir/tfile_%d' %c]

	kernel = miGraph.Kernel(svmGamma)
	kernel.read_data(N, fname_list)
	svmtrX = np.array(sorted([int(x.lstrip('tr')) for x in kernel.bags_list[0].keys()])).reshape(-1,1)
	svmvX = np.array(sorted([int(x.lstrip('v')) for x in kernel.bags_list[1].keys()])).reshape(-1,1)

	svmtrY = np.array([kernel.Y[kernel.bags_list[0]['tr%d'%x][0]] for x in svmtrX])

	kernel.compute_weight()
	print(c)

	clf = SVC(C=float(svmC), kernel=kernel.my_kernel)
	kernel.saved_dist = None
	clf.fit(svmtrX, svmtrY)

	kernel.saved_dist = None
	kernel.status = 2
	valid_pred[:,c] = miGraph.normalize(clf.decision_function(svmvX))

	kernel = None
	clf = None

# training
clist = np.logspace(-2, 2, 5)
gammalist = list(np.logspace(-3, 2, 4))

pool = ThreadPool(nthreads)

if os.path.isfile('dir/vccr.txt'):
	vccr = np.loadtxt('dir/vccr.txt')
else:
	vccr = np.zeros((len(clist), len(gammalist)))

for g0, svmGamma in enumerate(gammalist):
	#if g0 != dirnum:
	#	continue
	for c0, svmC in enumerate(clist):

		if vccr[c0,g0] != 0:
			continue

		valid_pred = np.zeros(vlbl.shape)
		runC = [c for c in range(C) if np.sum(trlbl[:,c])>0]	

		pool.map(train_model_class, runC)
		(roc, roc_macro) = myAUC.compute_auc(valid_pred, vlbl)
		vccr[c0,g0] = roc + np.random.randn()*1e-5 # tie breaker

		#vccr[c0,g0] = np.mean((vlbl==valid_pred)**2)
		print('>>>>', c0, g0, svmC, vccr[c0,g0])
		np.savetxt('dir/vccr.txt', vccr, '%f')


ind = np.unravel_index(vccr.argmax(), vccr.shape)
c0 = ind[0]
svmC = clist[c0]
g0 = ind[1]
svmGamma = gammalist[g0]


test_pred = np.zeros(tlbl.shape)

def test_model_class(c):

	global svmC
	global svmGamma
	global c0
	global test_pred

	fname_list = ['dir/trfile_%d' %c, 'dir/vfile_%d' %c, 'dir/tfile_%d' %c]

	
	kernel = miGraph.Kernel(svmGamma)
	kernel.read_data(N, fname_list)
	svmtrX = np.array(sorted([int(x.lstrip('tr')) for x in kernel.bags_list[0].keys()])).reshape(-1,1)
	svmtX = np.array(sorted([int(x.lstrip('t')) for x in kernel.bags_list[2].keys()])).reshape(-1,1)

	svmtrY = np.array([kernel.Y[kernel.bags_list[0]['tr%d'%x][0]] for x in svmtrX])

	kernel.compute_weight()

	clf = SVC(C=float(svmC), kernel=kernel.my_kernel)
	kernel.saved_dist = None
	clf.fit(svmtrX, svmtrY)

	print('test',c)
	kernel.saved_dist = None
	kernel.status = 3
	test_pred[:,c] = miGraph.normalize(clf.decision_function(svmtX))

pool.map(test_model_class, [c for c in range(C)])
(roc, roc_macro) = myAUC.compute_auc(test_pred, tlbl)

# ThFprAUC for documents with no labels
nolbld = np.where(np.sum(tlbl,1)==0)[0]
if len(nolbld) > 0:
	TH = np.linspace(0,1,50)
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

