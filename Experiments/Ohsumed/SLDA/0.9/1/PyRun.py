import numpy as np
import os, re, sys
from sklearn import metrics

sys.path.insert(0, '/'.join(os.getcwd().split('/')[:-4]))
import myAUC

dirnum = int(os.getcwd().split('/')[-1])-1
prop = os.getcwd().split('/')[-2]

load = 0

seed0 = 1000001 + 100*dirnum
np.random.seed(seed0)

dirpath = 'dir'
os.system('mkdir -p %s' %dirpath)
resfile = 'results.txt' 
fpres = open(resfile, 'w')
fpres.write('M roc wrdlkh roc_macro roc_trans roc_macro_trans, fprAUC traintime\n')
fpres.close()

Code = '../../../../../Code/SLDA/SSLDA'
Datapath = '../../../../../Data/Ohsumed'

trfile = '%s/lda_train-data.dat' %Datapath
if (prop != '1'):
        trlblfile = '%s/train-label%s.dat' %(Datapath,prop)
else:
        trlblfile = '%s/train-label.dat' %Datapath
tfile = '%s/lda_test-data.dat' %Datapath
tlblfile = '%s/test-label.dat' %Datapath
#vfile = '%s/valid-data.dat' %Datapath
#vlblfile = '%s/valid-label.dat' %Datapath
vocabfile = '%s/vocabs.txt' %Datapath

trlbl = np.loadtxt(trlblfile)
tlbl = np.loadtxt(tlblfile)
(Dtr, C) = trlbl.shape
Dt = tlbl.shape[0]
N = len(open(vocabfile).readlines())
alpha = 0.1
nu = 0.01
psi = [2.0, 2.0]
alpha_sigma = 0.3
nu_sigma = 0.3
M = 70
T = 100
BurnIn = 1500
ITER = 5
converged = 1e-3


# training 
settingfile = '%s/settings.txt' %dirpath
fp = open(settingfile, 'w')
fp.write('M %d\nC %d\nD %d\nN %d\nT %d\nalpha %f\nconverged %f' %(M,C,Dtr,N,T,alpha, converged))
fp.close()
seed = np.random.randint(seed0)

if load == 0:
	cmdtxt = '%s %d train %s %s %s random %s' %(Code, seed, trfile, trlblfile, settingfile, dirpath)
else:
	cmdtxt = '%s %d train %s %s %s load %s %s/001' %(Code, seed, trfile, trlblfile, settingfile, dirpath, dirpath)
print(cmdtxt)
os.system(cmdtxt)# + ' > /dev/null')

# transductive learning:
unlbld = np.where(trlbl[:,0]==-1)[0]
if len(unlbld) > 0:
	b = np.loadtxt('%s/final.b' %dirpath)[unlbld,:]
	gtlbl_unlbld = np.loadtxt('%s/train-label.dat' %Datapath)[unlbld,:]
	(roc_trans, roc_macro_trans) = myAUC.compute_auc(b, gtlbl_unlbld)
else:
	roc_trans = 0
	roc_macro_trans = 0

trtime = np.loadtxt('%s/likelihood.dat' %dirpath)[-1,2]

# test 
settingfile = '%s/settings.txt' %dirpath
fp = open(settingfile, 'w')
fp.write('M %d\nC %d\nD %d\nN %d\nT %d\nalpha %f\nconverged %f' %(M,C,Dt,N,T,alpha, converged))
fp.close()
seed = np.random.randint(seed0)

cmdtxt = '%s %d test %s %s %s/final %s' %(Code, seed, tfile, settingfile, dirpath, dirpath)
os.system(cmdtxt)# + ' > /dev/null')

wrdlkh = np.loadtxt('%s/test-lhood.dat' %dirpath)[1]

# class prediction
tlbl = np.loadtxt(tlblfile, dtype = np.int)
ypred = np.loadtxt('%s/testfinal.b' %dirpath)
(roc, roc_macro) = myAUC.compute_auc(ypred, tlbl)

# ThFprAUC for documents with no labels
nolbld = np.where(np.sum(tlbl,1)==0)[0]
if len(nolbld) > 0:
	TH = np.linspace(0,1,50)
	fpr = np.zeros(len(TH))
	for t,th in enumerate(TH):
		pred = np.round(ypred[nolbld] > th)
		tn = np.sum((1-pred) == 1)
		fp = np.sum(pred == 1)
		fpr[t] = fp/float(fp+tn)
	fprAUC = metrics.auc(TH,fpr)
else:
	fprAUC = 0

fpres = open(resfile, 'a')
fpres.write('%d %f %f %f %f %f %f %f\n' %(M, roc, wrdlkh, roc_macro, roc_trans, roc_macro_trans, fprAUC, trtime))
fpres.close()		

os.system('rm -r dir')

