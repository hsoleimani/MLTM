import re
import numpy as np

ifp = open('test-sentlabel.dat', 'r')
ofp = open('labeled_test_sentences.dat', 'w')


d = 0
while True:

    doc = ifp.readline()
    if len(doc) == 0:
        break
    
    sents = re.findall('<(.*?)>', doc)

    for s, sent in enumerate(sents):

        temp = np.array([float(x) for x in sent.split()])
        if temp[0] == -1:
            continue
        ofp.write('%d %d %s\n' %(d, s, sent))

    d += 1

ifp.close()
ofp.close()

