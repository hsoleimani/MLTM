import numpy as np
import os, re, sys
from sklearn import metrics
sys.path.insert(0, '/'.join(os.getcwd().split('/')[:-4]))
import myAUC

dirnum = int(os.getcwd().split('/')[-1])-1
prop = os.getcwd().split('/')[-2]

seed0 = 1000001 + 100*dirnum
np.random.seed(seed0)

code_path = '../../../../../Code/LR/MultiLabelMixLR'
Datapath = '../../../../../Data/Delicious'

trfile = '%s/train-data.dat' %Datapath
if prop != '1':
	trlblfile = '%s/train-label%s.dat' %(Datapath,prop)
else:
	trlblfile = '%s/train-label.dat' %(Datapath)
tfile = '%s/test-data.dat' %Datapath
tlblfile = '%s/test-label.dat' %Datapath
vfile = '%s/valid-data.dat' %Datapath
vlblfile = '%s/valid-label.dat' %Datapath
vocabfile = '%s/vocabs.txt' %Datapath
tslblfile = '%s/test-sentlabel.dat' %Datapath


resfile = 'results.txt'
if True:
	temp = np.loadtxt('results.txt',skiprows=1)
	done_num = temp.shape[0]
else:
	fpres = open(resfile, 'w')
	fpres.write('K roc roc_sent roc_macro roc_sent_macro fprAUC traintime\n')
	fpres.close()


# only keep labeled docs
os.system('mkdir -p dir')
fpdoc_in = open(trfile)
fpdoc_out = open('dir/train-data.dat','w')
fplbl_in = open(trlblfile)
fplbl_out = open('dir/train-label.dat','w')
while True:
	docln = fpdoc_in.readline()
	lblln = fplbl_in.readline()
	if len(docln)==0:
		break
	l0 = int(lblln.split(' ')[0])
	if l0 == -1:
		continue
	fpdoc_out.write(docln)
	fplbl_out.write(lblln)
trfile = 'dir/train-data.dat'
trlblfile = 'dir/train-label.dat'
fpdoc_in.close()
fpdoc_out.close()
fplbl_in.close()
fplbl_out.close()


trlbl = np.loadtxt(trlblfile)
tlbl = np.loadtxt(tlblfile)
(Dtr, C) = trlbl.shape
Dt = tlbl.shape[0]
N = len(open(vocabfile).readlines())

tlbl = np.loadtxt(tlblfile)


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

			
model_file = 'dir/final.w'

for k0,K in enumerate([1, 2, 4, 6, 10, 15]):

	if k0 < done_num:
		continue

	fp = open('settings.txt', 'w')
	fp.write('K %d\nC %d\nD %d\nN %d' %(K,C,Dtr,N))
	fp.close()

	# train
	seed = np.random.randint(seed0)
	cmdtxt = '%s %d train %s %s settings.txt random dir' %(code_path, seed, trfile, trlblfile)
	os.system(cmdtxt)

	trtime = np.loadtxt('dir/likelihood.dat')[-1,3]
	w = []
	for c in range(C):
		wtemp = np.loadtxt(model_file+str(c))
		w.append(wtemp)

	# do class prediction
	b_sent = np.zeros((cnt_lbld,C))
	scnt = 0
	scnt_lbld = 0
	b = np.zeros(tlbl.shape)
	fp = open(tfile)
	d = 0
	while True:
		ln = fp.readline()
		if len(ln)==0:
			break
		sents = re.findall('<[0-9]*?>([0-9 ]*)',ln)
		Sd = len(sents)-1
		py = np.zeros((Sd,C))
		for c in range(C):	
			logpy = 0.0;
			for s,sent in enumerate(sents[1:]):
				wTx = np.zeros(K)
				for n in sent.split():
					if K == 1:
						wTx[:] += w[c][int(n)]
					else:
						wTx[:] += w[c][int(n),:]

				maxval = np.max(wTx)
				if maxval > 0:
					temp = np.sum(np.exp(wTx - maxval))
					logpy -= maxval + np.log(np.exp(-maxval)+temp)
					py[s,c] = temp/(np.exp(-maxval)+temp)
				else:
					temp = np.sum(np.exp(wTx))
					logpy -= np.log(1+temp)
					py[s,c] = temp/(1.0+temp)

			b[d,c] = 1-np.exp(logpy)
		for s in range(Sd):
			if scnt not in slbld_list:
				scnt += 1
				continue
			b_sent[scnt_lbld,:] = py[s,:]
			scnt_lbld += 1
			scnt += 1
		d += 1
	fp.close()

	(roc, roc_macro) = myAUC.compute_auc(b, tlbl)
	(roc_sent, roc_sent_macro) = myAUC.compute_auc(b_sent, gt_sent)

	# ThFprAUC for documents with no labels
	nolbld = np.where(np.sum(tlbl,1)==0)[0]
	if len(nolbld) > 0:
		TH = np.linspace(0,1,20)
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
	fpres.write('%d %f %f %f %f %f %f\n' %(K,roc, roc_sent, roc_macro, roc_sent_macro, fprAUC, trtime))
	fpres.close()
	

os.system('rm -r dir')

