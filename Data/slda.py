# convert data to lda-c format (for slda)
import re
#trfile = 'Synthetic/docs.txt'
#tfile = 'Synthetic/test_docs.txt'
trfile = 'DBpedia/train-data.dat'
tfile = 'DBpedia/test-data.dat'

# train
fpin = open(trfile)
fname = '/'.join(trfile.split('/')[:-1]) + '/lda_' + trfile.split('/')[-1]
fpout = open(fname, 'w')
while True:
	ln = fpin.readline()
	if len(ln)==0:
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

# test
fpin = open(tfile)
fname = '/'.join(tfile.split('/')[:-1]) + '/lda_' + tfile.split('/')[-1]
fpout = open(fname, 'w')
while True:
	ln = fpin.readline()
	if len(ln)==0:
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
