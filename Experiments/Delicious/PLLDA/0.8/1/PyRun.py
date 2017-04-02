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
resfile = 'results.txt' 
fpres = open(resfile, 'w')
fpres.write('M roc roc_sent wrdlkh roc_macro roc_sent_macro incons_lblng fprAUC traintime\n')
fpres.close()

Code = '../../../../../Code/PLLDA/PLLDA'
Datapath = '../../../../../Data/Delicious'
lda_path = '../../../../../Code/lda/lda'

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
ofile = '%s/lda_obsvd-data.dat' %Datapath
hfile = '%s/lda_hldout-data.dat' %Datapath


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
alpha = 0.1
nu = 0.01
psi = [1.0, 1.0]
M = 10
T = 10000
BurnIn = 2000
ITER = 5
alpha_sigma = 0.2
nu_sigma = 0.1


# total number of sentences
Sd_total = 0
fp = open(tfile)
while True:
	ln = fp.readline()
	if len(ln)==0:
		break
	sd = re.findall('^<([0-9]*)>', ln)[0]
	Sd_total += int(sd)
fp.close()
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
Sd_total = cnt_lbld


slbl_pred = open(tfile).readlines()

fpout = open('%s/docs.txt' %dirpath, 'w')
for ln in slbl_pred:
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

fpout.close()


for tpc_per_lbl in [1,2,3,4]:
	for general in [0,1]:

		# is there any doc with no label?
		if (np.any(np.sum(np.loadtxt(trlblfile, dtype = np.int),1)==0)) and (general == 0):
			print('Skipping model with no general topics')
			continue

		# training 
		settingfile = '%s/settings.txt' %dirpath
		fp = open(settingfile, 'w')
		fp.write('general %d\nM %d\nC %d\nD %d\nN %d\nT %d\nBURNIN %d\n' %(general,(C+general)*tpc_per_lbl,C,Dtr,N,T,BurnIn))
		fp.write('alpha %f\nnu %f\npsi %f %f\nalpha_sigma %f\nnu_sigma %f' %(alpha, nu, psi[0], psi[1], alpha_sigma, nu_sigma))
		fp.close()
		seed = np.random.randint(seed0)

		cmdtxt = '%s %d train %s %s %s random %s' %(Code, seed, trfile, trlblfile, settingfile, dirpath)
		os.system(cmdtxt + ' > /dev/null')
		
		lbls_not_used = 100*np.loadtxt('%s/likelihood.dat' %dirpath)[-1,-1]

		trtime = np.loadtxt('%s/likelihood.dat' %dirpath)[-1,2]

		# test
		model_file = '%s/final.beta' %dirpath
		beta = np.loadtxt(model_file)
		beta /= np.sum(beta,0)
		(N,M) = beta.shape
		MCalpha = float(open('%s/final.alpha' %dirpath).read())
		np.savetxt('%s/lda.beta' %dirpath, np.log(beta).T)
		fp = open('%s/lda.other' %dirpath, 'w')
		fp.write('num_topics %d\nnum_terms %d\nalpha %f' %(M,N,MCalpha))
		fp.close()


		# run LDA
		seed = np.random.randint(seed0)
		cmdtxt = '%s %d inf ldasettings.txt %s/lda %s/docs.txt %s/test' \
			%(lda_path, seed, dirpath, dirpath, dirpath)
		print(cmdtxt)
		os.system(cmdtxt)

		# load LDA gamma
		theta = np.loadtxt('%s/test-gamma.dat' %dirpath)
		theta /= np.sum(theta,1).reshape(-1,1)

		# compute wrd-lkh
		fpdocs = open('%s/docs.txt' %dirpath)
		d = 0
		wrdlkh = 0.0
		while True:
			doc = fpdocs.readline()
			if len(doc) == 0:
				break
			wrds = re.findall('([0-9]*):[0-9]*', doc)
			cnts = re.findall('[0-9]*:([0-9]*)', doc)	
			doclkh = 0.0
			for n,w in enumerate(wrds):
				doclkh += float(cnts[n])*np.log(np.dot(theta[d,:], beta[int(w),:]))
			wrdlkh += doclkh
			d += 1
		fpdocs.close()

		# do class prediction
		tlbl = np.loadtxt(tlblfile, dtype = np.int)
		lbl_of_tpc = np.loadtxt('%s/final.tpc_lbl' %dirpath, dtype=np.int)

		TH = np.linspace(0,1+1e-4,50)
		fpr = np.zeros(len(TH))
		tpr = np.zeros(len(TH))
		fpr_macro = np.zeros((len(TH),C))
		tpr_macro = np.zeros((len(TH),C))
		for t,th in enumerate(TH):
			pred = np.zeros(tlbl.shape)
			for d in range(tlbl.shape[0]):
				actv_classes = [lbl_of_tpc[j] for j in np.where(theta[d,:]>th)[0] if lbl_of_tpc[j]<C]
				#print(actv_classes)
				pred[d, actv_classes] = 1
			tp = np.sum(tlbl*pred == 1)
			tn = np.sum((1-tlbl)*(1-pred) == 1)
			fp = np.sum(pred*(1-tlbl) == 1)
			fn = np.sum(tlbl*(1-pred) == 1)
			tpr[t] = tp/float(tp+fn)
			fpr[t] = fp/float(fp+tn)

			tp_macro = np.sum(tlbl*pred == 1,0)
			tn_macro = np.sum((1-tlbl)*(1-pred) == 1,0)
			fp_macro = np.sum(pred*(1-tlbl) == 1,0)
			fn_macro = np.sum(tlbl*(1-pred) == 1,0)
			tpr_macro[t,:] = tp_macro.astype(float)/(tp_macro+fn_macro)
			fpr_macro[t,:] = fp_macro.astype(float)/(fp_macro+tn_macro)


		roc = metrics.auc(fpr,tpr)
		roc_macroC = np.zeros(C)
		for c in range(C):
			roc_macroC[c] = metrics.auc(fpr_macro[:,c],tpr_macro[:,c])
		roc_macro = np.mean(roc_macroC)

	

		# ThFprAUC for documents with no labels
		nolbld = np.where(np.sum(tlbl,1)==0)[0]
		if len(nolbld) > 0:
			TH = np.linspace(0,1,50)
			fpr = np.zeros(len(TH))
			for t,th in enumerate(TH):
				pred = np.zeros((len(nolbld),C))
				for d0, d in enumerate(nolbld):
					actv_classes = [lbl_of_tpc[j] for j in np.where(theta[d,:]>th)[0] if lbl_of_tpc[j]<C]
					#print(actv_classes)
					pred[d0, actv_classes] = 1
				tn = np.sum((1-pred) == 1)
				fp = np.sum(pred == 1)
				fpr[t] = fp/float(fp+tn)
			fprAUC = metrics.auc(TH,fpr)
		else:
			fprAUC = 0


		### sentence prediction
		theta_sent = np.zeros((Sd_total,M))
		cnt = 0
		cnt_lbld = 0
		for d,doc in enumerate(slbl_pred):

			sents = re.findall('<[0-9]*?>([0-9 ]*)', doc)

			for s in range(len(sents)-1):
				if cnt not in slbld_list:
					cnt += 1
					continue
				wrds = sents[s+1].split()
				temp = np.zeros(M)
				for w in wrds:
					pp = theta[d,:]*beta[int(w),:]
					temp += pp/np.sum(pp)
				temp /= float(len(wrds))
				theta_sent[cnt_lbld,:] = temp
				cnt_lbld += 1
				cnt += 1


		TH = np.linspace(0,1+1e-4,50)
		fpr = np.zeros(len(TH))
		tpr = np.zeros(len(TH))
		fpr_macro = np.zeros((len(TH),C))
		tpr_macro = np.zeros((len(TH),C))
		for t,th in enumerate(TH):
			#pred = np.round(b_sent >= th)
			pred = np.zeros((Sd_total,C), dtype = np.int)
			for d in range(Sd_total):
				actv_classes = [lbl_of_tpc[j] for j in np.where(theta_sent[d,:]>th)[0] if lbl_of_tpc[j]<C]
				#print(actv_classes)
				pred[d, actv_classes] = 1
			tp = np.sum(gt_sent*pred == 1)
			tn = np.sum((1-gt_sent)*(1-pred) == 1)
			fp = np.sum(pred*(1-gt_sent) == 1)
			fn = np.sum(gt_sent*(1-pred) == 1)
			tpr[t] = tp/float(tp+fn)
			fpr[t] = fp/float(fp+tn)
		
			tp_macro = np.sum(gt_sent*pred == 1,0)
			tn_macro = np.sum((1-gt_sent)*(1-pred) == 1,0)
			fp_macro = np.sum(pred*(1-gt_sent) == 1,0)
			fn_macro = np.sum(gt_sent*(1-pred) == 1,0)
			tpr_macro[t,:] = tp_macro.astype(float)/(tp_macro+fn_macro)
			fpr_macro[t,:] = fp_macro.astype(float)/(fp_macro+tn_macro)

		roc_sent = metrics.auc(fpr,tpr)
		roc_macroC = np.zeros(C)
		nC = 0
		for c in range(C):
			if np.sum(gt_sent[:,c])==0:
				continue;
			roc_macroC[c] = metrics.auc(fpr_macro[:,c],tpr_macro[:,c])
			nC += 1
		roc_sent_macro = np.sum(roc_macroC)/float(nC)

		# compute wrd-lkh on the heldout set
		# run LDA
		seed = np.random.randint(seed0)
		cmdtxt = '%s %d inf ldasettings.txt %s/lda %s %s/test' \
			%(lda_path, seed, dirpath, ofile, dirpath)
		print(cmdtxt)
		os.system(cmdtxt)

		# load LDA gamma
		theta = np.loadtxt('%s/test-gamma.dat' %dirpath)
		theta /= np.sum(theta,1).reshape(-1,1)

		# compute wrd-lkh
		fpdocs = open(hfile)
		d = 0
		wrdlkh = 0.0
		while True:
			doc = fpdocs.readline()
			if len(doc) == 0:
				break
			wrds = re.findall('([0-9]*):[0-9]*', doc)
			cnts = re.findall('[0-9]*:([0-9]*)', doc)	
			doclkh = 0.0
			for n,w in enumerate(wrds):
				doclkh += float(cnts[n])*np.log(np.dot(theta[d,:], beta[int(w),:]))
			wrdlkh += doclkh
			d += 1
		fpdocs.close()

		fpres = open(resfile, 'a')
		fpres.write('%d %f %f %f %f %f %f %f %f\n' %(M, roc, roc_sent, wrdlkh, roc_macro, roc_sent_macro, lbls_not_used, fprAUC, trtime))
		fpres.close()

os.system('rm -r dir')
