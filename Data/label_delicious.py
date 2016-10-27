'''
This code is used for manually labeling sentences in delicious/test-data.dat
'''
import re
import numpy as np
from termcolor import colored
#np.random.seed(100001)

def read_sent_labels(fname, C):

	fp = open(fname)
	d = 0
	sent_lbls = dict()
	n_lbld_sent = 0
	while True:
		ln = fp.readline()
		if len(ln) == 0:
			break
		sents = re.findall('<(.*?)>',ln)
		temp = []
		for sent in sents:
			tt = [int(x) for x in sent.split()]
			if (0 in tt) or (1 in tt):
				n_lbld_sent += 1
			temp.append(tt)
		sent_lbls.update({d:temp})
		d += 1
	fp.close()
	return sent_lbls,n_lbld_sent

def write_sent_labels(fname, sent_lbls):
	
	fp = open(fname, 'w')

	for d in range(len(sent_lbls)):
		txt = ""
		for sent in sent_lbls[d]:
			txt += "<" + " ".join([str(x) for x in sent]) + ">"
		fp.write(txt + '\n')			

	return None

lblfilename = 'Delicious/test-sentlabel.dat'
# read raw_sentences
fpin = open('Delicious/test-rawdata.dat','r')
sent_breaker = re.compile(r'<s>(.*?)</s>')
sent_txt = {}
d = 0
while True:
	ln = fpin.readline()
	if len(ln) == 0:
		break
	doc = sent_breaker.findall(ln)
	sent_txt.update({d:[x for x in doc]})
	d += 1
fpin.close()

# read label strings:
lblname = open('Delicious/label_list.dat').readline().split(', ')
C = len(lblname)

# read doc-level labels
doclbl = np.loadtxt('Delicious/test-label.dat',dtype=np.int)


fresh = raw_input('Fresh start (y/n)? ')

if fresh == 'y':
	temp = raw_input('CONFIRM: THIS WILL ERASE ALL SENTENCES ALREADY LABELED (y/n)? ')
	if temp == 'n':
		sent_lbls,n_lbld_sent = read_sent_labels(lblfilename, C)
	else:
		sent_lbls = dict()
		for d in range(len(sent_txt)):
			Sd = len(sent_txt[d])
			temp = []
			for s in range(Sd):
				temp.append([-1 for x in range(C)])
			sent_lbls.update({d:temp})
			d += 1
		n_lbld_sent = 0
		
else:
	sent_lbls,n_lbld_sent = read_sent_labels(lblfilename, C)


# start labeling:
Dt = len(sent_txt)
redo_chk = 0
session_cnt = 0
while True:

	d = np.random.choice(Dt, 1)[0]


	if np.all(doclbl[d] == 0):
		continue

	Sd = len(sent_txt[d])

	# print doclbls and the sentence
	print("---------------------------------------------------------------")
	txt = []
	c0 = 1
	doc_c = {}
	for c in np.where(doclbl[d]==1)[0]:
		txt.append(lblname[c]+' (%d)' %c0)
		doc_c.update({c0:c})
		c0 += 1
	print(', '.join(txt) )
	for s,sent in enumerate(sent_txt[d]):
		print "%d) %s"%(s, sent),\
			colored(','.join([lblname[c+1] for c in range(C) if sent_lbls[d][s][c]==1]),"green"),\
			colored(','.join([lblname[c] for c in range(C) if sent_lbls[d][s][c]==0]),"yellow")

	exit_chk = 0
	while True:

		inp = raw_input('sent_id labels > ' )

		# read in input
		if inp == 'exit':
			exit_chk = 1
			break
		elif inp == "":
			break
		#elif inp == 'redo': # query previous sentence again
		#	redo_chk = 1
		inp_split = inp.split()
		s = int(inp_split[0])
		if sent_lbls[d][s][0] != -1:
			n_lbld_sent += 1
		for c in range(C):
			sent_lbls[d][s][c] = 0
		if len(inp_split)>1:
			if inp_split[1] == '-1':
				for c in range(C):
					sent_lbls[d][s][c] = -1
			else:
				for c0 in inp_split[1:]:
					c = doc_c[int(c0)]
					sent_lbls[d][s][c] = 1

		prev_d = d
		prev_s = s

	if exit_chk == 1:
		break
	session_cnt += 1
	if session_cnt%2 == 0:
		print (n_lbld_sent)
		write_sent_labels(lblfilename, sent_lbls)

print("number of labeled sentences: %d" %n_lbld_sent)
write_sent_labels(lblfilename, sent_lbls)


