import re, os
import numpy as np
import urllib2
from itertools import groupby
from nltk.stem.porter import *
import nltk
stemmer = PorterStemmer()
np.random.seed(100000001)
 
# download and extract http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.tar.gz
os.system('mkdir -p Reuters')
request = urllib2.urlopen('http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.tar.gz')
output = open("Reuters/reuters21578.tar.gz","w")
output.write(request.read())
output.close()
os.system('mkdir -p Reuters/data')
os.system('tar -zvxf Reuters/reuters21578.tar.gz -C Reuters/data')


# download and read stopword list
import urllib2
request = urllib2.urlopen('http://www.textfixer.com/resources/common-english-words.txt')
output = open("Reuters/stopwords.txt","w")
output.write(request.read())
output.close()

stpfile = open('Reuters/stopwords.txt', 'r')
stpwrdlist = stpfile.readlines()[0].split(',')
stpfile.close()
stpwrds = set(stpwrdlist)
stpwrds.add('reuter')


# crawl all folders
lbl_list = []
lbl_freq = []
C = 0
rawvocabs = set()
documents = list() 
doc_lbls = list()
num_doc_lbl = list()
# some reg exp patterns
email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}')
nonchar = re.compile(r'[^a-zA-Z]')
# read training docs
directory = 'Reuters/data/'
d = 0
for path, dirs, files in os.walk(directory):
	dirs.sort(reverse = False)

	for f in files:
		filename = os.path.join(path, f)
		print(filename)
		if 'reut2' not in filename:
			continue

		# read file
		fp = open(filename)
		fileraw = fp.read()
		fp.close()
		docstxt = fileraw.split('</REUTERS>')
		
		for docline in docstxt:

			if '<TOPICS>' not in docline:
				continue
			if '<TEXT TYPE="BRIEF">' in docline: # breif docs don't have text body
				continue

			# extract topic(s)
			topicstring = docline.split('<TOPICS>')[1].split('</TOPICS>')[0]
			topics = re.findall('<D>([a-zA-Z-]*)</D>',topicstring)
			if len(topics)==0: # skip documents with no labels
				if np.random.random() > 0.3:
					continue
			for lbl in topics:
				try:
					lbl_num = lbl_list.index(lbl)
					lbl_freq[lbl_num] += 1
				except ValueError:
					lbl_list.append(lbl)
					lbl_freq.append(1)
					C += 1

			topictxt = '|'.join(topics)

			# extract body and title
			if '<BODY>' not in docline:
				continue
			maintxt = docline.split('<BODY>')[1].split('</BODY>')[0] 
			if '<TITLE>' in docline:
				maintxt += ' ' + docline.split('<TITLE>')[1].split('</TITLE>')[0]

			maintxt = re.sub(r'\n', ' ', maintxt)	# replace \n with space

			doc_temp = []
			sents = nltk.sent_tokenize(maintxt)
			for sent in sents:
				#maintxt = re.sub(r'\n', ' ', sent)   # replace \n with space
				maintxt = nonchar.sub(" ", sent)
				lntxt = [x.lower() for x in maintxt.split() if len(x)>=3 and x.lower() not in stpwrds] #remove words with <= 2
				if len(lntxt) <= 3: # skipping sentences with <3 words
					continue
				rawvocabs = rawvocabs.union(set(lntxt))
				doc_temp.extend(['%s' %(' '.join(lntxt))])

			documents.extend(['%s' %('|'.join(doc_temp))])
			doc_lbls.extend(['%s' %topictxt])
			d += 1

			#if d>100:
			#	break
		#break
	#break
print('Done with reading docs; %d classes, %d docs' %(C,len(doc_lbls)))

# only keep top 20 classes
top_classes = np.argsort(np.array(lbl_freq))[-20:]
c_keep = [lbl_list[c] for c in range(C) if c in top_classes] 
lbl_list = list(c_keep)

doc_lbls_2 = list()
for doclbl in doc_lbls:
	temp = [x for x in doclbl.split('|') if x in c_keep]
	doc_lbls_2.extend(['|'.join(temp)])
doc_lbls = list(doc_lbls_2)

print('Make train/test/valid splits')
train_ind = []
valid_ind = []
test_ind = []
ind = np.arange(len(doc_lbls))								
num = int(len(ind)*0.65)
trind = np.random.choice(ind, num, replace = False)
left = np.setdiff1d(ind, trind)
num = int(len(left)*0.1)
vind = np.random.choice(left, num, replace = False)
tind = np.setdiff1d(left, vind)
train_ind.extend(list(trind))
valid_ind.extend(list(vind))
test_ind.extend(list(tind))

'''print('Creating stemmed vocab list')    
N = 0
vocabs = {}
stem_to_vocabs = {}
trrawvocabs = set()
for d in train_ind:
	trrawvocabs = trrawvocabs.union(set(re.split(r'\|| ', documents[d])))

for w in trrawvocabs:
	if w not in stem_to_vocabs.keys():
		sw = stemmer.stem(w)
		try:
			wnum = vocabs[sw]	
		except KeyError:
			wnum =  N
			vocabs.update({sw:N})
			N += 1
		stem_to_vocabs.update({w:wnum})
for w in rawvocabs:
	if w not in stem_to_vocabs.keys():
		sw = stemmer.stem(w)
		try:
			wnum = vocabs[sw]	
			stem_to_vocabs.update({w:wnum})
		except KeyError:
			continue
'''
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
fp = open('Reuters/vocabs.txt','w')
for vv in sorted(vocabs.items(), key = operator.itemgetter(1)):
	fp.write('%s, %d\n' %(vv[0], vv[1]))
fp.close()
print('total words: %d, total training words: %d, final vocabs(stemmed): %d, final words: %d'\
	%(len(rawvocabs), len(trrawvocabs), len(vocabs), len(stem_to_vocabs)))

print('Writing docs')
fptr = open('Reuters/train-data.dat','w')
fptrlbl = open('Reuters/train-label.dat','w')
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
		if ls > 0:
			doc_txt += ' <%d> %s' %(ls, ' '.join(sent_wrds))
			Sd += 1
			nw += ls
	doc_txt = '<%d>' %Sd + doc_txt
	if Sd < 3:
		continue
	#print(doc_lbls[d])
	fptr.write(doc_txt+'\n')
	fptrlbl.write('%s\n' %' '.join(['1' if x in doc_lbls[d].split('|') else '0' for x in lbl_list]))
fptr.close()
fptrlbl.close()

fp = open('Reuters/label_list.dat','w')
fp.write(', '.join(lbl_list))
fp.close()


fpt = open('Reuters/test-data.dat','w')
fpv = open('Reuters/valid-data.dat','w')
fpt_lbl = open('Reuters/test-label.dat','w')
fpv_lbl = open('Reuters/valid-label.dat','w')
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
		if ls > 0:
			doc_txt += ' <%d> %s' %(ls, ' '.join(sent_wrds))
			Sd += 1
			nw += ls
	doc_txt = '<%d>' %Sd + doc_txt
	if Sd < 3:
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


# delete downloaded/extracted files
os.system('rm -rf Reuters/data')
os.system('rm Reuters/reuters21578.tar.gz')
