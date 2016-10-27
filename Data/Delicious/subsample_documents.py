import re
import numpy as np

np.random.seed(1000001)

source_location = '/home/studadmin/Desktop/delicioust140_documents/fdocuments'
docnamefile = 'DocNameTag.txt'
outdocfile = 'Subsampled_DocNameTag.txt'


Classes = set(['reference', 'design', 'programming', 'internet', 'computer', 'web', 'java', 'writing', 'english', 'grammar', 'style', 'language', 'books', 'education', 'philosophy', 'politics', 'religion', 'science', 'history','culture'])

doc_w_lbl = {}

docnames = open(docnamefile).readlines()

doc_lbls = list()
doc_fnames = list()
numlbls = list()
d = 0;
for ln in docnames:
	stuff = re.findall('<(.*?)>', ln)
	fname = stuff[0]
	tags = stuff[1].split('|')
	topics = [x for x in tags if x in Classes]
	#fname = '%s/%s/%s' %(source_location,fname[:2],fname)
	if '.html' not in fname:
		continue
	numlbls.append(len(topics))
	doc_lbls.extend(['%s' %('|'.join(topics))])
	doc_fnames.extend([fname])
	for x in topics:
		try:
			doc_w_lbl[x].append(d)
		except KeyError:
			doc_w_lbl.update({x:[d]})
	d += 1
	if d%5000 == 0:
		print d

numlbls = np.array(numlbls)

# sample 1100 from each lbl
all_ind = set()
for x in Classes:
	prob = numlbls[doc_w_lbl[x]]/float(np.sum(numlbls[doc_w_lbl[x]]))
	ind = np.random.choice(doc_w_lbl[x], min(1100, len(doc_w_lbl[x])), p = prob, replace = False)
	all_ind = all_ind.union(set(ind))
# sample 1000 docs with no label
ind = np.random.choice(np.where(numlbls == 0)[0], 1000, replace = False)
all_ind = all_ind.union(set(ind))

all_ind = list(all_ind)

fp = open(outdocfile, 'w')
for d in all_ind:
	fp.write('<%s><%s>\n' %(doc_fnames[d], doc_lbls[d]))
fp.close()

