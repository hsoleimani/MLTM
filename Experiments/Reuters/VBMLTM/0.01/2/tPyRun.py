import numpy as np
import os, re, sys
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, '/'.join(os.getcwd().split('/')[:-4]))
import myAUC

dirnum = int(os.getcwd().split('/')[-1])-1
prop = os.getcwd().split('/')[-2]

seed0 = 1001 + 100*dirnum
np.random.seed(seed0)

dirpath = 'dir'

os.system('mkdir -p %s' %dirpath)

Code = '../../../../../Code/MLTMVB/MLTMVB'
LDACode = '../../../../../Code/LDA_VB_Parallel/lda_vb'
Datapath = '../../../../../Data/Reuters' 
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
tslblfile = '%s/test-sentlabel.dat' %Datapath
ofile = '%s/obsvd-data.dat' %Datapath
hfile = '%s/lda_hldout-data.dat' %Datapath

resfile = 'results.txt' 
fpres = open(resfile, 'w')
fpres.write('roc wrdlkh roc_macro roc_trans roc_macro_trans prob_lbling_error fprAUC traintime\n')
fpres.close()


trlbl = np.loadtxt(trlblfile)
tlbl = np.loadtxt(tlblfile)
(Dtr, C) = trlbl.shape
Dt = tlbl.shape[0]
N = len(open(vocabfile).readlines())
#N = 500
M = 70
alpha = 0.1#1.0/float(M)
nu = 0.01
kappa = 0.6#51 #0.9
tau = 2000.0
T = 300000
L = 10000
batchsize = 500
lag = 10000#50
save_lag = 5000
psi1, psi2 = 1.0, 1.0
rho0 = 0.1
BurnIn = 50
s = 10

trtime = np.loadtxt('%s/likelihood.dat' %dirpath)[-1,3]

# transductive learning:
unlbld = np.where(trlbl[:,0]==-1)[0]
if len(unlbld) > 0:
	b = np.loadtxt('%s/001.b' %dirpath)[unlbld,:]
	gtlbl_unlbld = np.loadtxt('%s/train-label.dat' %Datapath)[unlbld,:]
	(roc_trans, roc_macro_trans) = myAUC.compute_auc(b, gtlbl_unlbld)
else:
	roc_trans = 0
	roc_macro_trans = 0

# test
tsettingfile = '%s/test_settings.txt' %dirpath
fp = open(tsettingfile, 'w')
fp.write('M %d\nC %d\nD %d\nN %d\nT %d\nL %d\n' %(M,C,Dt,N,T,L))
fp.write('alpha %f\nnu %f\nkappa %f\ntau %f\nbatchsize %d\n' %(alpha, nu, kappa, tau, batchsize))
fp.write('lag %d\nsave_lag %d\npsi %f %f\nrho0 %f\nburnin %d\ns %f\n' %(lag, save_lag, psi1, psi2, rho0, BurnIn,s))
fp.close()


seed = np.random.randint(seed0)

cmdtxt = '%s %d test %s %s %s/001 %s' %(Code,seed,tfile, tsettingfile, dirpath, dirpath)
#os.system(cmdtxt)
os.system(cmdtxt  + ' > /dev/null')

wrdlkh = np.loadtxt('%s/test-lhood.dat' %dirpath)[-3]
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


# compute probability of labeling error
prob_lbling_error = 0.0
cnt = 0.0
slbl_pred = open('%s/001.y' %dirpath).readlines()
for doc in slbl_pred:
	sents_pred = doc.split('|')[:-1]
	pys = np.zeros((len(sents_pred), C))
	for s, sent in enumerate(sents_pred):
		pys[s, :] = np.array([float(x) for x in sent.split()])
	A = np.max(pys, 0)
	B = 1.0 - np.prod(1-pys, 0)
	prob_lbling_error += np.sum((1.0 - A) * B) + np.sum(A * (1.0 - B))
	cnt += 1.0
prob_lbling_error /= cnt * float(C)



### compute lkh on the heldout test set
seed = np.random.randint(seed0)
cmdtxt = '%s %d test %s %s %s/001 %s' %(Code, seed, ofile, tsettingfile, dirpath, dirpath)
os.system(cmdtxt  + ' > /dev/null')

theta = np.loadtxt('%s/testfinal.theta' %dirpath)
theta /= np.sum(theta, 1).reshape(-1, 1)
beta = np.loadtxt('%s/001.beta' %dirpath)
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
fpres.write('%f %f %f %f %f %f %f %f\n' %(roc, wrdlkh, roc_macro, roc_trans, roc_macro_trans, prob_lbling_error, fprAUC, trtime))
fpres.close()		


#os.system('rm dir/00*')
os.system('rm dir/test*')


