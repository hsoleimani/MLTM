import numpy as np
import os, re, sys
from sklearn import metrics

sys.path.insert(0, '/'.join(os.getcwd().split('/')[:-4]))
import myAUC

dirnum = int(os.getcwd().split('/')[-1])-1
prop = os.getcwd().split('/')[-2]

seed0 = 10001 + 100*dirnum
np.random.seed(seed0)

dirpath = 'dir'

os.system('mkdir -p %s' %dirpath)

Code = '../../../../../Code/MLTM/MultiLabelTM'
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
fpres.write('roc wrdlkh roc_macro roc_trans roc_macro_trans fprAUC traintime\n')
fpres.close()

trtime = np.loadtxt('%s/likelihood.dat' %dirpath)[-1,2]


# transductive learning:
unlbld = np.where(trlbl[:,0]==-1)[0]
roc_trans = 0
roc_macro_trans = 0
if len(unlbld) > 0: 
	gtlbl_unlbld = np.loadtxt('%s/train-label.dat' %Datapath)[unlbld,:]
	if os.path.isfile('%s/final.b' %dirpath):
		b = np.loadtxt('%s/final.b' %dirpath)[unlbld,:]
		(roc_trans, roc_macro_trans) = myAUC.compute_auc(b, gtlbl_unlbld)
	elif os.path.isfile('%s/001.b' %dirpath):
		b = np.loadtxt('%s/001.b' %dirpath)[unlbld,:]
		(roc_trans, roc_macro_trans) = myAUC.compute_auc(b, gtlbl_unlbld)




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

if os.path.isfile('%s/final.b' %dirpath):
	cmdtxt = '%s %d test %s %s %s/final %s' %(Code,seed,tfile, tsettingfile, dirpath, dirpath)
else:
	cmdtxt = '%s %d test %s %s %s/001 %s' %(Code,seed,tfile, tsettingfile, dirpath, dirpath)
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


fpres = open(resfile, 'a')
fpres.write('%f %f %f %f %f %f %f\n' %(roc, wrdlkh, roc_macro, roc_trans, roc_macro_trans, fprAUC, trtime))
fpres.close()		

#os.system('rm -r dir')
