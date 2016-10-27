import re, nltk, os
from html2text import html2text
import numpy as np
import urllib2
from itertools import groupby
from nltk.stem.porter import *

stemmer = PorterStemmer()
np.random.seed(100000001)

# keeping only 10 classes
Allclasses = ['Company', 'EducationalInstitution', 'Artist', 'Athlete',  
	'MeanOfTransportation', 'Building', 'NaturalPlace', 'Village', 'Animal',
	 'Film'] 
# reads data from /DBpedia/documents.txt

# download and read stopword list
import urllib2
request = urllib2.urlopen('http://www.textfixer.com/resources/common-english-words.txt')
output = open("DBpedia/stopwords.txt","w")
output.write(request.read())
output.close()

stpfile = open('DBpedia/stopwords.txt', 'r')
stpwrdlist = stpfile.readlines()[0].split(',')
stpfile.close()
stpwrds = set(stpwrdlist)

C = len(Allclasses)
rawvocabs = set()
orig_documents = list() 
orig_doc_lbls = list()
num_doc_lbl = np.zeros(C)
class_doc_ind = [list() for x in range(C)]
nonchar = re.compile(r'[^a-zA-Z]')
lbl_pattern = re.compile(r'^([^ ]*) ')

filename = 'DBpedia/documents.txt'
fp = open(filename)
d = 0
while True:
	ln = fp.readline()
	if len(ln) == 0:
		break
	lbl = lbl_pattern.findall(ln)[0]
	maintxt = lbl_pattern.split(ln)[-1]
	
	try:
		lbl_num = Allclasses.index(lbl)
	except ValueError:
		continue

	num_from_this_lbl = num_doc_lbl[lbl_num]
	if (num_from_this_lbl >= 5000) or (np.random.random() > 0.1):
		continue

	sents = nltk.sent_tokenize(maintxt)
	doc_temp = []
	for sent in sents:

		maintxt = nonchar.sub(" ", sent).lower()
		lntxt = [x for x in maintxt.split() if len(x)>=3 and nonchar.search(x)==None  and x not in stpwrds] #remove words with <= 2
		rawvocabs = rawvocabs.union(set(lntxt))
		doc_temp.extend(['%s' %(' '.join(lntxt))])
	
	orig_documents.append(doc_temp)
	orig_doc_lbls.append(lbl_num)
	num_doc_lbl[lbl_num] += 1.0
	class_doc_ind[lbl_num].append(d)
	d += 1
	if d%1000 == 0:
		print d
		
	#if d>2000:
	#	break
print('Done with reading docs; %d classes, %d docs' %(C,len(orig_doc_lbls)))

## create documents by mix and match from original docs
D = 8000
doc_lbls = list()
documents = list()
class_prob = np.array(num_doc_lbl)
class_prob /= np.sum(class_prob)

for d in range(D):
	# choose num of sentences
	Sd = np.random.poisson(7) + 1

	# choose documents until have >Sd sents
	doc_pool = []
	lbl_pool = []
	pool_sd = 0
	while True:
		while len(lbl_pool)<C:
			c0 = np.random.choice(C,1,p=class_prob)[0]
			if c0 not in lbl_pool:
				break
		d0 = np.random.choice(class_doc_ind[c0], 1)[0]
		doc_pool.append(d0)
		lbl_pool.append(c0)
		pool_sd += len(orig_documents[d0])
		if pool_sd > 1.5*Sd and len(lbl_pool)>1:
			break
	# generate the doc
	
	doc_temp = []
	lbl_temp = []
	for sd in range(Sd):
		d0 = np.random.choice(doc_pool,1)[0]
		c0 = orig_doc_lbls[d0]
		s0 = np.random.choice(len(orig_documents[d0]),1)
		doc_temp.append(orig_documents[d0][s0])
		lbl_temp.append(str(c0))
	documents.extend(['%s' %('|'.join(doc_temp))])
	doc_lbls.extend(['%s' %('|'.join(lbl_temp))])
		
	if d%1000 == 0:
		print d
	
	
	

print('Make train/test/valid splits')
train_ind = []
valid_ind = []
test_ind = []
ind = np.arange(len(doc_lbls))
#keep_ind = np.random.choice(ind, int(len(ind)*0.85), replace = False)
#ind = keep_ind.copy() 												
num = 4500#int(len(ind)*0.65)
trind = np.random.choice(ind, num, replace = False)
left = np.setdiff1d(ind, trind)
num = 3000#int(len(left)*0.1)
vind = np.random.choice(left, num, replace = False)
tind = np.setdiff1d(left, vind)
train_ind.extend(list(trind))
valid_ind.extend(list(vind))
test_ind.extend(list(tind))

print('Creating stemmed vocab list')    
N = 0
vocabs = {}
stem_to_vocabs = {}
trrawvocabs = set()
for d in train_ind:
	trrawvocabs = trrawvocabs.union(set(re.split(r'\|| ', documents[d])))

for w in trrawvocabs:
	#if w not in stem_to_vocabs.keys():
	sw = stemmer.stem(w)
	try:
		wnum = vocabs[sw]	
	except KeyError:
		wnum =  N
		vocabs.update({sw:N})
		N += 1
	#stem_to_vocabs.update({w:wnum})

# compute word counts (to remove less frequent ones)
vocabcnts = np.zeros(N)
for d in train_ind:
	docraw = documents[d]
	raw_sentences = docraw.split('|')
	for sent in raw_sentences:
		sent_wrds = []
		for wrd in sent.split():
			try:
				w = stemmer.stem(wrd)#stem_to_vocabs[wrd]
				vocabcnts[vocabs[w]] += 1.0
			except KeyError:
				continue

# remove words with <= 3 occurrences
N = 0
vocabs_reduced = {}
import operator
sorted_dic = sorted(vocabs.items(), key=operator.itemgetter(1)) # sort by wrd index
for vpair in sorted_dic[1:]: # wrd 0 is ''; skipping that; don't know where it came from
	w = vpair[0]
	n_old = vpair[1]
	if vocabcnts[n_old] <= 3:
		continue
	vocabs_reduced.update({w:N})
	N += 1 
vocabs = vocabs_reduced.copy()

for w in rawvocabs:
	if w not in stem_to_vocabs.keys():
		sw = stemmer.stem(w)
		try:
			wnum = vocabs[sw]	
			stem_to_vocabs.update({w:wnum})
		except KeyError:
			continue



print('Writing dictionary')
import operator
fp = open('DBpedia/vocabs.txt','w')
for vv in sorted(vocabs.items(), key = operator.itemgetter(1)):
	fp.write('%s, %d\n' %(vv[0], vv[1]))
fp.close()
print('total words: %d, total training words: %d, final vocabs(stemmed): %d, final words: %d'\
	%(len(rawvocabs), len(trrawvocabs), len(vocabs), len(stem_to_vocabs)))

print('Writing docs')
fptr = open('DBpedia/train-data.dat','w')
fptrlbl = open('DBpedia/train-label.dat','w')
fptrsentlbl = open('DBpedia/train-sentlabel.dat','w')
train_lbls = []
for d in train_ind:
	docraw = documents[d]
	lblraw = [int(x) for x in doc_lbls[d].split('|')]
	lbl_union = set(lblraw)
	doclbl_txt = ' '.join(['1' if x in lbl_union else '0' for x in range(C)])
	raw_sentences = docraw.split('|')
	Sd = 0
	nw = 0
	doc_txt = ''
	sentlbl_txt = ''
	for s,sent in enumerate(raw_sentences):
		sent_wrds = []
		for wrd in sent.split():
			try:
				w = str(stem_to_vocabs[wrd])
				sent_wrds.append(w)
			except KeyError:
				continue
		ls = len(sent_wrds)
		if ls > 2:
			doc_txt += ' <%d> %s' %(ls, ' '.join(sent_wrds))
			Sd += 1
			nw += ls
			sentlbl_txt += '<%s>' %' '.join(['1' if x==lblraw[s] else '0' for x in range(C)])
	doc_txt = '<%d>' %Sd + doc_txt
	if Sd < 2:
		continue
	fptr.write(doc_txt+'\n')
	fptrlbl.write('%s\n' %doclbl_txt)
	fptrsentlbl.write('%s\n' %sentlbl_txt)
fptr.close()
fptrlbl.close()
fptrsentlbl.close()


fpt = open('DBpedia/test-data.dat','w')
fpv = open('DBpedia/valid-data.dat','w')
fpt_lbl = open('DBpedia/test-label.dat','w')
fpv_lbl = open('DBpedia/valid-label.dat','w')
fpt_sentlbl = open('DBpedia/test-sentlabel.dat','w')
fpv_sentlbl = open('DBpedia/valid-sentlabel.dat','w')
for d, doc in enumerate(documents):
	if d in train_ind:
		continue

	docraw = documents[d]
	lblraw = [int(x) for x in doc_lbls[d].split('|')]
	lbl_union = set(lblraw)
	doclbl_txt = ' '.join(['1' if x in lbl_union else '0' for x in range(C)])
	raw_sentences = docraw.split('|')
	Sd = 0
	nw = 0
	doc_txt = ''
	sentlbl_txt = ''
	for s,sent in enumerate(raw_sentences):
		sent_wrds = []
		for wrd in sent.split():
			try:
				w = str(stem_to_vocabs[wrd])
				sent_wrds.append(w)
			except KeyError:
				continue
		ls = len(sent_wrds)
		if ls > 2:
			doc_txt += ' <%d> %s' %(ls, ' '.join(sent_wrds))
			Sd += 1
			nw += ls
			sentlbl_txt += '<%s>' %' '.join(['1' if x==lblraw[s] else '0' for x in range(C)])
	doc_txt = '<%d>' %Sd + doc_txt
	if Sd < 2:
		continue

	if d in test_ind:
		fpt.write(doc_txt+'\n')
		fpt_lbl.write('%s\n' %doclbl_txt)
		fpt_sentlbl.write('%s\n' %sentlbl_txt)
	elif d in valid_ind:
		fpv.write(doc_txt+'\n')
		fpv_lbl.write('%s\n' %doclbl_txt)
		fpv_sentlbl.write('%s\n' %sentlbl_txt)
	
fpt.close()
fpv.close()
fpt_lbl.close()
fpv_lbl.close()
fpt_sentlbl.close()
fpv_sentlbl.close()


