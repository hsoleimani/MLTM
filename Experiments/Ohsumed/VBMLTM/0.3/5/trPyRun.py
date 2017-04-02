import numpy as np
import os, re, sys
sys.path.insert(0,'/gpfs/home/hus152/anaconda2/lib/python2.7/site-packages/')
#sys.path.insert(0, '/'.join(os.getcwd().split('/')[:-4]))

dirnum = int(os.getcwd().split('/')[-1])-1
prop = os.getcwd().split('/')[-2]

seed0 = 1001 + 100*dirnum
np.random.seed(seed0)

dirpath = 'dir'

os.system('mkdir -p %s' %dirpath)

Code = '../../../../../Code/MLTMVB/MLTMVB'
LDACode = '../../../../../Code/LDA_VB_Parallel/lda_vb'
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
tslblfile = '%s/test-sentlabel.dat' %Datapath

resfile = 'results.txt' 
#fpres = open(resfile, 'w')
#fpres.write('roc roc_sent wrdlkh roc_macro roc_sent_macro roc_trans roc_macro_trans fprAUC traintime\n')
#fpres.close()


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

settingfile = '%s/settings.txt' %dirpath
fp = open(settingfile, 'w')
fp.write('M %d\nC %d\nD %d\nN %d\nT %d\nL %d\n' %(M,C,Dtr,N,T,L))
fp.write('alpha %f\nnu %f\nkappa %f\ntau %f\nbatchsize %d\n' %(alpha, nu, kappa, tau, batchsize))
fp.write('lag %d\nsave_lag %d\npsi %f %f\nrho0 %f\nburnin %d\ns %f\n' %(lag, save_lag, psi1, psi2, rho0, BurnIn,s))
fp.close()


# write docs in LDA format
os.system('mkdir -p dir')
ldadoc = 'dir/docs.txt'
fpin = open(trfile)
fpout = open(ldadoc, 'w')
while True:
	ln = fpin.readline()
	if len(ln) == 0:
		break
	sents = re.findall('<[0-9]*?>([0-9 ]*)',ln)
	wrds = []
	cnts = []
	for sent in sents[1:]:
		for w in sent.split():
			try:
				i = wrds.index(w)
				cnts[i] += 1
			except ValueError:
				wrds.append(w)
				cnts.append(1)
	txt = ' '.join(['%s:%s' %(w, cnts[i]) for i,w in enumerate(wrds)])
	fpout.write('%d %s\n' %(len(wrds), txt))	
fpin.close()
fpout.close()


# run lda
s1 = np.random.randint(seed0)
cmdtxt = LDACode + ' ' + str(s1) + ' est ' + ldadoc + ' ' + str(M) + ' seeded dir/vblda'
cmdtxt += ' ' + str(alpha) + ' ' + str(nu)
print(cmdtxt)
os.system(cmdtxt)
os.system('cp dir/vblda/final.beta dir/init.beta')

fp = open('dir/init.w', 'w')
for c in range(C):
	txt = ['%f' %x for x in np.random.uniform(0.475,0.525,M)]
	fp.write(' '.join(txt)+'\n')
fp.close()

os.system('rm -r dir/vblda/')
os.system('rm dir/docs.txt')


seed = np.random.randint(seed0)
cmdtxt = '%s %d train %s %s %s loadtopics %s %s/init' %(Code, seed, trfile, trlblfile, settingfile, dirpath, dirpath)
#cmdtxt = '%s %d train %s %s %s seeded %s' %(Code, seed, trfile, trlblfile, settingfile, dirpath)
os.system(cmdtxt)

	
