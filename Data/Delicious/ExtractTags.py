#http://nlp.uned.es/social-tagging/delicioust140/
import re, os
import numpy as np

# This file reads taginfo.xml, and extract filenames and >=2 common tags for each document.

tagfile = 'delicioust140_taginfo/taginfo.xml'
outfile = 'DocNameTag.txt'
TagCntFile = 'TagCnts.txt'

fp = open(tagfile)
TagsXml = fp.read()
fp.close()

fp = open(outfile, 'w')
TagCnt = {}

documents = re.findall(r'<document>(.*?)</document>', TagsXml, re.DOTALL)

for d,doc in enumerate(documents):
	num_users = float(re.findall('<users>([0-9]*)</users>',doc)[0])
	tagsTxt = re.findall('<tag>(.*?)</tag>',doc,re.DOTALL)
	docname = re.findall('<filename>(.*?)</filename>', doc)[0]
	doc_tags = []
	for tagNum, tagTxt in enumerate(tagsTxt):
		tag = re.findall('<name><\!\[CDATA\[(.*?)\]\]></name>',tagTxt)[0]
		weight = float(re.findall('<weight>(.*?)</weight>',tagTxt)[0])
		if (tagNum > 1) and (weight/num_users < 0):
			break
		doc_tags.append(tag)
		try:
			TagCnt[tag] += 1.0
		except KeyError:
			TagCnt.update({tag:1.0})
	fp.write('<%s><%s>\n' %(docname, '|'.join(doc_tags)))
	#print(doc_tags)
	if d%10000 == 0:
		print(d, len(TagCnt))
	
fp.close()

import operator
sorted_dic = sorted(TagCnt.items(), key=operator.itemgetter(1),reverse=True)
fp = open(TagCntFile, 'w')
for item in sorted_dic:
	fp.write('%s %d\n' %(item[0], int(item[1])))
fp.close()
