import numpy as np
import os, re, sys
from sklearn import metrics

sys.path.insert(0, '/'.join(os.getcwd().split('/')[:-4]))
import myAUC

dirnum = int(os.getcwd().split('/')[-1])-1
prop = os.getcwd().split('/')[-2]

seed0 = 1000001 + 100*dirnum
np.random.seed(seed0)

dirpath = 'dir'

os.system('mkdir -p %s' %dirpath)

Code = '../../../../../Code/MLTM/MultiLabelTM'
Datapath = '../../../../../Data/DBpedia'
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
trslblfile = '%s/train-sentlabel.dat' %Datapath
tslblfile = '%s/test-sentlabel.dat' %Datapath
ofile = '%s/obsvd-data.dat' %Datapath
hfile = '%s/lda_hldout-data.dat' %Datapath

trlbl = np.loadtxt(trlblfile)
tlbl = np.loadtxt(tlblfile)
(Dtr, C) = trlbl.shape
Dt = tlbl.shape[0]
N = len(open(vocabfile).readlines())
#N = 500
T = 10000
BurnIn = 2000

resfile = 'results.txt' 
fpres = open(resfile, 'w')
fpres.write('roc roc_sent wrdlkh roc_macro roc_sent_macro roc_trans roc_macro_trans fprAUC traintime\n')
fpres.close()


# total number of sentences
Sd_total = 0
fp = open(tfile)
while True:
	ln = fp.readline()
	if len(ln)==0:
		break
	sd = re.findall('^<([0-9]*)>', ln)[0]
	Sd_total += int(sd)

slbl_gt = open(tslblfile).readlines()
gt_sent = np.zeros((Sd_total,C))
cnt = 0
for d in range(len(slbl_gt)):
	sents_gt = re.findall('<(.*?)>',slbl_gt[d])
	for sent in sents_gt:
		gt_sent[cnt,:] = np.array([float(x) for x in sent.split()])
		cnt += 1

trtime = np.loadtxt('%s/likelihood.dat' %dirpath)[-1,2]


# transductive learning:
unlbld = np.where(trlbl[:,0]==-1)[0]
if len(unlbld) > 0:
	b = np.loadtxt('%s/final.b' %dirpath)[unlbld,:]
	gtlbl_unlbld = np.loadtxt('%s/train-label.dat' %Datapath)[unlbld,:]
	(roc_trans, roc_macro_trans) = myAUC.compute_auc(b, gtlbl_unlbld)
else:
	roc_trans = 0
	roc_macro_trans = 0



# test
tsettingfile = '%s/test_settings.txt' %dirpath
trsettingfile = '%s/settings.txt' %dirpath
fpt = open(tsettingfile, 'w')
fptr = open(trsettingfile, 'r')
ln_num = 0
while True:
	ln = fptr.readline()
	if len(ln) == 0:
		break
	if ln_num == 2:
		fpt.write('D %d\n' %Dt)
	else:
		fpt.write(ln)
	ln_num += 1
fpt.close()
fptr.close()

seed = np.random.randint(seed0)

cmdtxt = '%s %d test %s %s %s/final %s' %(Code,seed,tfile, tsettingfile, dirpath, dirpath)
#print(cmdtxt)
os.system(cmdtxt  + ' > /dev/null')

wrdlkh = np.loadtxt('%s/test-lhood.dat' %dirpath)[-1,1] 
b = np.loadtxt('%s/testfinal.b' %dirpath)

(roc, roc_macro) = myAUC.compute_auc(b, tlbl)

# ThFprAUC for documents with no labels
nolbld = np.where(np.sum(tlbl,1)==0)[0]
if len(nolbld) > 0:
	TH = np.linspace(0,1,50)
	fpr = np.zeros(len(TH))
	for t,th in enumerate(TH):
		pred = np.round(b[nolbld] > th)
		tn = np.sum((1-pred) == 1)
		fp = np.sum(pred == 1)
		fpr[t] = fp/float(fp+tn)
	fprAUC = metrics.auc(TH,fpr)
else:
	fprAUC = 0


#sentence prediction
slbl_pred = open('%s/testfinal.MCy' %dirpath).readlines()
b = np.zeros((Sd_total,C))
cnt = 0
for doc in slbl_pred:
	sents_pred = doc.split('|')[:-1]
	for sent in sents_pred:
		b[cnt,:] = np.array([float(x) for x in sent.split()])
		cnt += 1

(roc_sent, roc_sent_macro) = myAUC.compute_auc(b, gt_sent)

### compute lkh on the heldout test set
seed = np.random.randint(seed0)
cmdtxt = '%s %d test %s %s %s/final %s' %(Code, seed, ofile, tsettingfile, dirpath, dirpath)
os.system(cmdtxt  + ' > /dev/null')

theta = np.loadtxt('%s/testfinal.theta' %dirpath)
theta /= np.sum(theta, 1).reshape(-1, 1)
beta = np.loadtxt('%s/final.beta' %dirpath)
beta /= np.sum(beta, 0)
# hldout
fp = open(hfile)
wrdlkh = 0.0
d = 0
while True:
	doc = fp.readline()
	if len(doc) == 0:
		break
	wrds = [int(x) for x in re.findall('([0-9]*):[0-9]*', doc)]
	cnts = [float(x) for x in re.findall('[0-9]*:([0-9]*)', doc)]
	tmp = np.log(np.dot(beta[wrds, :], theta[d,:].reshape(-1,1)))[:,0]
	wrdlkh += np.sum(tmp.copy()*np.array(cnts))
	d += 1
fp.close()


fpres = open(resfile, 'a')
fpres.write('%f %f %f %f %f %f %f %f %f\n' %(roc, roc_sent, wrdlkh, roc_macro, roc_sent_macro, roc_trans, roc_macro_trans, fprAUC, trtime))
fpres.close()	

os.system('rm -r dir')
