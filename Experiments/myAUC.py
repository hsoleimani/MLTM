import numpy as np
from sklearn import metrics

def compute_auc(b, tlbl, npts=50):
	C = b.shape[1]
	TH = np.linspace(0,1+1e-4,npts)
	fpr = np.zeros(len(TH))
	tpr = np.zeros(len(TH))
	fpr_macro = np.zeros((len(TH),C))
	tpr_macro = np.zeros((len(TH),C))
	for t,th in enumerate(TH):
		pred = np.round(b >= th)
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
		tpr_macro[t,:] = (tp_macro.astype(float)+1e-10)/(tp_macro+fn_macro+1e-10)
		fpr_macro[t,:] = fp_macro.astype(float)/(fp_macro+tn_macro)

	roc = metrics.auc(fpr,tpr)

	roc_macro = np.zeros(C)
	nC = 0.0
	for c in range(C):
		if np.sum(tlbl[:,c],0)==0:
			continue
		roc_macro[c] = metrics.auc(fpr_macro[:,c],tpr_macro[:,c])
		nC += 1.0

	return(roc, np.sum(roc_macro)/nC)

