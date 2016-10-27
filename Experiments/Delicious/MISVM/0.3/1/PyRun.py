import numpy as np
import os, re, sys
#sys.path.insert(0,'/gpfs/home/hus152/.local/lib/python3.3/site-packages/')
sys.path.insert(0,'/gpfs/home/hus152/anaconda2/lib/python2.7/site-packages/')
from sklearn import metrics
from multiprocessing.dummy import Pool as ThreadPool 
from sklearn.externals import joblib

sys.path.append('/'.join(os.getcwd().split('/')[:-4]))
import myAUC

sys.path.append('../../../../../Code/MIL')
from MISVM import *
from testSVM import *


dirnum = int(os.getcwd().split('/')[-1])-1
prop = os.getcwd().split('/')[-2]

nthreads = int(sys.argv[1])

seed0 = 1000001 + 100*dirnum
np.random.seed(seed0)

dirpath = 'dir'
os.system('mkdir -p %s' %dirpath)
resfile = 'results.txt' 
fpres = open(resfile, 'w')
fpres.write('roc roc_sent roc_macro roc_sent_macro fprAUC\n')
fpres.close

Datapath = '../../../../../Data/Delicious'

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

# total number of sentences
Sd_total = 0
fp = open(sent_tfile)
while True:
	ln = fp.readline()
	if len(ln)==0:
		break
	sd = re.findall('^<([0-9]*)>', ln)[0]
	Sd_total += int(sd)

slbl_gt = open(tslblfile).readlines()
gt_sent2 = np.zeros((Sd_total,C))
cnt = 0
cnt_lbld = 0
slbld_list = []
for d in range(len(slbl_gt)):
	sents_gt = re.findall('<(.*?)>',slbl_gt[d])
	for sent in sents_gt:
		temp = np.array([float(x) for x in sent.split()])
		if temp[0] == -1:
			cnt += 1
			continue
		gt_sent2[cnt_lbld,:] = temp.copy()
		slbld_list.append(cnt)
		cnt_lbld += 1
		cnt += 1
gt_sent = gt_sent2[:cnt_lbld,:].copy()

# training
clist = np.logspace(-2, 2, 5)
vccr = np.zeros(len(clist))

def train_model_class(c):
	global svmC
	global c0
	global valid_pred
	
	if os.path.isfile('dir/tmp_%d_%d.tar.gz' %(c,c0)):
		os.system('tar -zxf dir/tmp_%d_%d.tar.gz ' %(c,c0))
		clf = joblib.load('dir/tmp_%d_%d/model.pkl' %(c,c0))
	else:
		(bag_pred_f, ypred_f, clf, bags) = MISVM(N, 'dir/trfile_'+str(c), svmC)
		#save model
		os.system('mkdir -p dir/tmp_%d_%d' %(c,c0))
		joblib.dump(clf, 'dir/tmp_%d_%d/model.pkl' %(c,c0))

	# validation
	(bag_pred_f, ypred_f, bags) = testSVM(N, 'dir/vfile_%d' %c, clf)

	valid_pred[:,c] = np.round(bag_pred_f)
	if not os.path.isfile('dir/tmp_%d_%d.tar.gz' %(c,c0)):
		os.system('tar -zcf dir/tmp_%d_%d.tar.gz dir/tmp_%d_%d' %(c,c0,c,c0))
	os.system('rm -r dir/tmp_%d_%d' %(c,c0))


pool = ThreadPool(nthreads) 
for c0, svmC in enumerate(clist):

	runC = [c for c in range(C) if np.sum(trlbl[:,c])>0]
	valid_pred = np.zeros(vlbl.shape)
	# write the mfiles
	#for c in range(C):
	pool.map(train_model_class, runC)
	vccr[c0] = np.mean((vlbl==valid_pred)**2)
	print('>>>>',c0,svmC,vccr[c0])

c0 = np.argmax(vccr)
bestC = clist[c0]


# make prediction on the test set
test_pred = np.zeros(tlbl.shape)
temp_sent = np.zeros(gt_sent2.shape)
test_sent_pred = np.zeros(gt_sent.shape)
svmC = bestC
#for c in range(C):
def test_model_class(c):
	global test_pred
	global test_sent_pred
	global temp_sent
	global svmC
	global c0
	global slbld_list

	os.system('tar -zxf dir/tmp_%d_%d.tar.gz ' %(c,c0))
	clf = joblib.load('dir/tmp_%d_%d/model.pkl' %(c,c0))

	(test_pred[:,c], temp_sent[:,c], bags) = testSVM(N, 'dir/tfile_%d' %c, clf)
	test_sent_pred[:,c] = temp_sent[slbld_list,c].copy()

pool.map(test_model_class, [c for c in range(C)])

(roc, roc_macro) = myAUC.compute_auc(test_pred, tlbl)

(roc_sent, roc_sent_macro) = myAUC.compute_auc(test_sent_pred, gt_sent)


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
fpres.write('%f %f %f %f %f\n' %(roc, roc_sent, roc_macro, roc_sent_macro, fprAUC))
fpres.close()	

os.system('rm -r dir')

