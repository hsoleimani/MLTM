import re, nltk, os
from html2text import html2text
import numpy as np
import urllib2
from itertools import groupby
from nltk.stem.porter import *

stemmer = PorterStemmer()
np.random.seed(100000001)

source_location = '/home/studadmin/Desktop/delicioust140_documents/fdocuments'
docnamefile = 'Delicious/Subsampled_DocNameTag.txt'

docnames = open(docnamefile).readlines()

stpfile = open('Delicious/stopwords.txt', 'r')
stpwrdlist = stpfile.readlines()[0].split(',')
stpfile.close()
stpwrds = set(stpwrdlist)

request = urllib2.urlopen('http://mirror.ctan.org/systems/win32/winedt/dict/english.zip')
output = open("Delicious/english.zip","w")
output.write(request.read())
output.close()
os.system('mkdir -p Delicious/english')
os.system('unzip -oq Delicious/english.zip -d Delicious/english')
Wrdslist = set()
for path, dirs, files in os.walk('Delicious/english'):
	dirs.sort(reverse = True)
	for f in files:
		if '.dic' not in f: 
			continue
		filename = os.path.join(path, f)
		fp = open(filename)
		temp = fp.readlines()
		fp.close()
		Wrdslist = Wrdslist.union([x.rstrip() for x in temp if '%' not in x])


lbl_list = []
C = 0
rawvocabs = set()
documents = list() 
doc_lbls = list()
num_doc_lbl = list()

nonchar = re.compile(r'[^a-zA-Z]')

from bs4 import BeautifulSoup

for d, ln in enumerate(docnames):
	#if d < 1400:
	#	continue
	stuff = re.findall('<(.*?)>', ln)
	fname = stuff[0]
	topics = stuff[1].split('|')
	fname = '%s/%s/%s' %(source_location,fname[:2],fname)
	if '.html' not in fname:
		continue
	rawhtml = open(fname).read()
	try:
		#rawtxt = html2text(rawhtml.decode('ascii','ignore').encode('ascii','ignore'))
		soup = BeautifulSoup(rawhtml.decode('ascii','ignore').encode('ascii','ignore'), 'html.parser')
		rawtxt = soup.get_text()
	except ValueError:
		continue
	sents = nltk.sent_tokenize(rawtxt)
	doc_temp = []
	for sent in sents:

		maintxt = nonchar.sub(" ", sent).lower()
		lntxt = [x for x in maintxt.split() if len(x)>=3 and x in Wrdslist and x not in stpwrds] #remove words with <= 2
		if len(lntxt) <= 3 or len(lntxt)>10:  # skipping sentences with <3 or >10 words
			continue

		if len(doc_temp) > 30: # was 50 
			break
		rawvocabs = rawvocabs.union(set(lntxt))
		doc_temp.extend(['%s' %(' '.join(lntxt))])

	for lbl in topics:
		if len(lbl)==0:
			continue
		try:
			lbl_num = lbl_list.index(lbl)
		except ValueError:
			lbl_list.append(lbl)
			C += 1
	documents.extend(['%s' %('|'.join(doc_temp))])
	doc_lbls.extend(['%s' %('|'.join(topics))])
	if d%100 == 0:
		print d
	#if d >= 200:
	#	break

 
print('Make train/test/valid splits')
train_ind = []
valid_ind = []
test_ind = []
ind = np.arange(len(doc_lbls))
keep_ind = np.random.choice(ind, int(len(ind)*0.85), replace = False)
ind = keep_ind.copy() 												
num = int(len(ind)*0.65)
trind = np.random.choice(ind, num, replace = False)
left = np.setdiff1d(ind, trind)
num = int(len(left)*0.1)
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
	if vocabcnts[n_old] <= 5:
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
fp = open('Delicious/vocabs.txt','w')
for vv in sorted(vocabs.items(), key = operator.itemgetter(1)):
	fp.write('%s, %d\n' %(vv[0], vv[1]))
fp.close()
print('total words: %d, total training words: %d, final vocabs(stemmed): %d, final words: %d'\
	%(len(rawvocabs), len(trrawvocabs), len(vocabs), len(stem_to_vocabs)))

print('Writing docs')
fptr = open('Delicious/train-data.dat','w')
fptrlbl = open('Delicious/train-label.dat','w')
train_lbls = []
for d in train_ind:
	docraw = documents[d]
	raw_sentences = docraw.split('|')
	Sd = 0
	nw = 0
	doc_txt = ''
	for sent in raw_sentences:
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
	doc_txt = '<%d>' %Sd + doc_txt
	if Sd < 2:
		continue
	print(doc_lbls[d])
	fptr.write(doc_txt+'\n')
	fptrlbl.write('%s\n' %' '.join(['1' if x in doc_lbls[d].split('|') else '0' for x in lbl_list]))
fptr.close()
fptrlbl.close()

fp = open('Delicious/label_list.dat','w')
fp.write(', '.join(lbl_list))
fp.close()


fpt = open('Delicious/test-data.dat','w')
fpv = open('Delicious/valid-data.dat','w')
fpt_lbl = open('Delicious/test-label.dat','w')
fpv_lbl = open('Delicious/valid-label.dat','w')
for d, doc in enumerate(documents):
	if d in train_ind:
		continue
	docraw = documents[d]
	docwrds = []
	raw_sentences = docraw.split('|')
	Sd = 0
	nw = 0
	doc_txt = ''
	for sent in raw_sentences:
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
	doc_txt = '<%d>' %Sd + doc_txt
	if Sd < 2:
		continue

	if d in test_ind:
		fpt.write(doc_txt+'\n')
		fpt_lbl.write('%s\n' %' '.join(['1' if x in doc_lbls[d].split('|') else '0' for x in lbl_list]))
	elif d in valid_ind:
		fpv.write(doc_txt+'\n')
		fpv_lbl.write('%s\n' %' '.join(['1' if x in doc_lbls[d].split('|') else '0' for x in lbl_list]))
	
fpt.close()
fpv.close()
fpt_lbl.close()
fpv_lbl.close()



