dname = 'umed'
nthrds = 4
for prop in ['0.1','0.3','0.6','0.8','0.9','0.05','0.01','1']:
	if float(prop)>0.9:
		nthrds = 8
	elif float(prop)>0.3:
		nthrds = 4
	else:
		nthrds = 2
	nthrds = 10 #***********************************
	for t in range(1,6):
		fname = 'tr%s_%s_%d' %(dname,prop,t)
		fp = open(fname,'w')
		fp.write("#PBS -l nodes=1:ppn=%d\n" %nthrds)
		fp.write("#PBS -l walltime=24:00:00\ncd $PBS_O_WORKDIR\n")
		fp.write("module load gsl\nmodule load python/3.3.2\n")
		fp.write("cd %s/%d\n" %(prop, t))
		fp.write("export OMP_NUM_THREADS=%d\n" %nthrds)
		fp.write("python3 trPyRun.py")
		fp.close()
		tnthrds = 2
		fname = 't%s_%s_%d' %(dname,prop,t)
		fp = open(fname,'w')
		fp.write("#PBS -l nodes=1:ppn=%d\n" %tnthrds)
		fp.write("#PBS -l walltime=24:00:00\ncd $PBS_O_WORKDIR\n")
		fp.write("module load gsl\nmodule load python/3.3.2\n")
		fp.write("cd %s/%d\n" %(prop, t))
		fp.write("export OMP_NUM_THREADS=%d\n" %tnthrds)
		fp.write("python3 tPyRun.py")
		fp.close()
