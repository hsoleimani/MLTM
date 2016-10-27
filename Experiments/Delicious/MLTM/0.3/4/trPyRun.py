import numpy as np
import os, re
from sklearn import metrics

dirnum = int(os.getcwd().split('/')[-1])-1
prop = os.getcwd().split('/')[-2]

seed0 = 1000001 + 100*dirnum
np.random.seed(seed0)

dirpath = 'dir'

os.system('mkdir -p %s' %dirpath)

Code = '../../../../../Code/MLTM/MultiLabelTM'
Datapath = '../../../../../Data/Delicious'
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
alpha = 0.1
nu = 0.01
psi = [2.0, 2.0]
M = 60
T = 10000
BurnIn = 2000
P = 10

settingfile = '%s/settings.txt' %dirpath
fp = open(settingfile, 'w')
fp.write('M %d\nC %d\nD %d\nN %d\nT %d\nburnin %d\n' %(M,C,Dtr,N,T,BurnIn))
fp.write('alpha %f\nnu %f\npsi %f %f\nsent 1\nP %d\n' %(alpha, nu, psi[0], psi[1],P))
fp.write('psi_sigma %f\nalpha_sigma %f\nnu_sigma %f\n' %(0.4,0.15,0.2))
fp.close()

fp.close()
seed = np.random.randint(seed0)

cmdtxt = '%s %d train %s %s %s random %s' %(Code, seed, trfile, trlblfile, settingfile, dirpath)
os.system(cmdtxt)

