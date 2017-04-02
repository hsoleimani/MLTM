import numpy as np
import os, re, sys

#print(sys.argv)
dataset = sys.argv[1]
method = 'VBMLTM'#sys.argv[2]

print('Processing %s method, %s dataset' %(method, dataset))

T = 5
#if method == 'PLLDA':
#	M = int(sys.argv[3])
directory = '%s/%s' %(dataset,method)
resfile = 'Results/%s_perr_%s.txt' %(dataset,method)
prop = [0.01, 0.05, 0.1, 0.3, 0.6, 0.8, 0.9, 1]
for pcnt,p in enumerate(prop):
	for t in range(T):
		path = '%s/%s/%s' %(directory,str(p),str(t+1)) 

		files = os.listdir(path)	
		for f in files:
			if 'prob_lbling_error.txt' not in f:
				continue
			filename = path + '/' + f
			temp_read = np.loadtxt(filename,skiprows=0)
			temp = temp_read.copy()			
			if pcnt == 0 and t == 0:
				results = np.zeros((len(prop),2*len(temp)+1))
			if t == 0:
				prop_res = np.zeros((T,len(temp)))
			prop_res[t,:] = temp
	results[pcnt,:] = np.hstack((p,np.mean(prop_res,0),np.std(prop_res,0)))
	#print(p, results[pcnt,:])

np.savetxt(resfile, results, '%f')
