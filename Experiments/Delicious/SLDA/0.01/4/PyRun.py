import numpy as np
import os, re, sys
from sklearn import metrics
from scipy.special import psi

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
fpres.write('roc roc_sent wrdlkh roc_macro roc_sent_macro roc_trans roc_macro_trans, fprAUC traintime\n')
fpres.close()

Code = '../../../../../Code/SLDA/SSLDA'
Datapath = '../../../../../Data/Delicious'

trfile = '%s/lda_train-data.dat' %Datapath
if (prop != '1'):
        trlblfile = '%s/train-label%s.dat' %(Datapath,prop)
else:
        trlblfile = '%s/train-label.dat' %Datapath
tfile = '%s/lda_test-data.dat' %Datapath
tlblfile = '%s/test-label.dat' %Datapath
ofile = '%s/lda_obsvd-data.dat' %Datapath
hfile = '%s/lda_hldout-data.dat' %Datapath
#vfile = '%s/valid-data.dat' %Datapath
#vlblfile = '%s/valid-label.dat' %Datapath
vocabfile = '%s/vocabs.txt' %Datapath
sent_tfile = '%s/test-data.dat' %Datapath
tslblfile = '%s/test-sentlabel.dat' %Datapath

trlbl = np.loadtxt(trlblfile)
tlbl = np.loadtxt(tlblfile)
(Dtr, C) = trlbl.shape
Dt = tlbl.shape[0]
N = len(open(vocabfile).readlines())
alpha = 0.1
nu = 0.01
alpha_sigma = 0.3
nu_sigma = 0.3
M = 60
T = 100
BurnIn = 1500
ITER = 5
converged = 1e-3



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
os.system(cmdtxt + ' > /dev/null')

wrdlkh = np.loadtxt('%s/test-lhood.dat' %dirpath)[1]

# class prediction
tlbl = np.loadtxt(tlblfile, dtype = np.int)
ypred = np.loadtxt('%s/testfinal.b' %dirpath)
(roc, roc_macro) = myAUC.compute_auc(ypred, tlbl)

# ThFprAUC for documents with no labels
nolbld = np.where(np.sum(tlbl,1)==0)[0]
if len(nolbld) > 0:
	TH = np.linspace(0,1,20)
	fpr = np.zeros(len(TH))
	for t,th in enumerate(TH):
		pred = np.round(ypred[nolbld] > th)
		tn = np.sum((1-pred) == 1)
		fp = np.sum(pred == 1)
		fpr[t] = fp/float(fp+tn)
	fprAUC = metrics.auc(TH,fpr)
else:
	fprAUC = 0

######## sentence prediction

# compute phi given gamma and do prediction
gamma = np.loadtxt('%s/testfinal.gamma' %dirpath) #[d,j]
logbeta = np.loadtxt('%s/final.beta' %dirpath) #[n,j]
eta = np.loadtxt('%s/final.w' %dirpath) #[j,c]

b = np.zeros((cnt_lbld,C))
cnt = 0
cnt_lbld = 0
fp = open(sent_tfile)
for d in range(Dt):
	ln = fp.readline()
	psi_gamma = psi(gamma[d,:]) - psi(np.sum(gamma[d,:]))
	sents = re.findall('<[0-9]*?>([0-9 ]*)',ln)
	for sent in sents[1:]:
		if cnt not in slbld_list:
			cnt += 1
			continue
		wrds = [int(w) for w in sent.split()]
		phi = psi_gamma + logbeta[wrds,:]
		phi = np.exp(phi - np.max(phi,1).reshape(-1,1))
		phi /= np.sum(phi,1).reshape(-1,1)
		ypred = np.exp(np.dot(np.mean(phi,0),eta))
		b[cnt_lbld,:] = ypred/(1+ypred)
		cnt_lbld += 1
		cnt += 1

fp.close()

(roc_sent, roc_sent_macro) = myAUC.compute_auc(b, gt_sent)

# compute test-set likelihood
# obsvd 
seed = np.random.randint(seed0)
cmdtxt = '%s %d test %s %s %s/final %s' %(Code, seed, ofile, settingfile, dirpath, dirpath)
print(cmdtxt)
os.system(cmdtxt)# + ' > /dev/null')

theta = np.loadtxt('%s/testfinal.gamma' %dirpath)
theta /= np.sum(theta, 1).reshape(-1, 1)
beta = np.exp(np.loadtxt('%s/final.beta' %dirpath))
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

