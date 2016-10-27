import numpy as np
import os, re, sys

#print(sys.argv)
dataset = sys.argv[1]
method = sys.argv[2]

print('Processing %s method, %s dataset' %(method, dataset))

T = 5
#if method == 'PLLDA':
#	M = int(sys.argv[3])
directory = '%s/%s' %(dataset,method)
resfile = 'Results/%s_%s.txt' %(dataset,method)
prop = [0.01, 0.05, 0.1, 0.3, 0.6, 0.8, 0.9, 1]
for pcnt,p in enumerate(prop):
	for t in range(T):
		path = '%s/%s/%s' %(directory,str(p),str(t+1)) 

		files = os.listdir(path)	
		for f in files:
			if 'result' not in f:
				continue
			filename = path + '/' + f
			temp_read = np.loadtxt(filename,skiprows=1)
			if (method in ['PLLDA','LR']):
				#ind = np.where(temp_read[:,0]==M)[0]
				#print(filename, temp_read)
				#if dataset == 'Ohsumed':
				#	ind = np.argmax(temp_read[:6,1])
				#else:
				ind = np.argmax(temp_read[:,1])
				temp = temp_read[ind,:].reshape(-1)
				#print(temp[0])
			else:
				temp = temp_read.copy()			
			if pcnt == 0 and t == 0:
				results = np.zeros((len(prop),2*len(temp)+1))
			if t == 0:
				prop_res = np.zeros((T,len(temp)))
			prop_res[t,:] = temp
		if method in ['MISVM','mi_SVM']:
			for tt in range(1,T):
				prop_res[tt,:] = prop_res[t,:]
			break
	results[pcnt,:] = np.hstack((p,np.mean(prop_res,0),np.std(prop_res,0)))
	#print(p, results[pcnt,:])

np.savetxt(resfile, results, '%f')
