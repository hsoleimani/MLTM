import numpy as np
import sys

prop = sys.argv[1]

for t in range(1,5):

	temp = np.loadtxt('%s/%d/dir/vccr.txt' %(prop,t))
	if t == 1:
		vccr = temp.copy()
	else:
		vccr[:,t-1] = temp[:,t-1]


np.savetxt('%s/1/dir/vccr.txt'%prop, vccr,'%f')

