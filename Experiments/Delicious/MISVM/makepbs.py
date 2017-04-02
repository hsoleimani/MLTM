dname = 'MIdeli'
for prop in ['0.1','0.3','0.6','0.8','0.9','0.05','0.01','1']:
	if float(prop)>=0.3:
		nthreads = 6
	else:
		nthreads = 4
	if float(prop)>=0.6:
		mem = 4
	else:
		mem = 2
	for t in range(1,6):
		fname = '%s_%s_%d' %(dname,prop,t)
		fp = open(fname,'w')
		fp.write("#PBS -l nodes=1:ppn=%d\n" %nthreads)
		fp.write("#PBS -l pmem=%dgb\n"%mem)
		fp.write("#PBS -l walltime=24:00:00\ncd $PBS_O_WORKDIR\n")
		fp.write("module unload python\n")
		#fp.write("module load python/3.3.2\n")
		fp.write("export OMP_NUM_THREADS=%d\n"%nthreads)
		fp.write("cd %s/%d\n" %(prop, t))
		fp.write("python PyRun.py %d" %nthreads) 
		fp.close()

		break
