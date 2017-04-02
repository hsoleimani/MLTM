# This script splits each test document in half. We estimate topic proportions
# on the first half (observed set) and compute likelihood on the second half (held-out set)

import re
import numpy as np

np.random.seed(1000001)

for dataset in ['Reuters', 'Ohsumed', 'Delicious', 'DBpedia']:

    print dataset

    fname = '%s/test-data.dat' %dataset

    fpin = open(fname)
    fp_obsvd = open('%s/obsvd-data.dat' %dataset, 'w')
    fp_hldout = open('%s/hldout-data.dat' %dataset, 'w')
    fp_lda_obsvd = open('%s/lda_obsvd-data.dat' %dataset, 'w')
    fp_lda_hldout = open('%s/lda_hldout-data.dat' %dataset, 'w')

    while True:
        ln = fpin.readline()
        if len(ln)==0:
            break
        sents = re.findall('(<[0-9]*?>[0-9 ]*)',ln)
        Sd = int(re.findall('<([0-9]+)>', sents[0])[0])

        oSd = int(Sd/2)
        hSd = Sd - oSd
        fp_obsvd.write('<%d>' %(oSd))
        fp_hldout.write('<%d>' %(hSd))

        o_sent_ids = np.random.choice(Sd, oSd, replace=False)

        o_wrds = []
        o_cnts = []
        h_wrds = []
        h_cnts = [] 
        for sid, sent in enumerate(sents[1:]):
            if sid in o_sent_ids:
                fp_obsvd.write(' ' + sent)
                for w in sent.split('> ')[1].split():
                    try:
                        i = o_wrds.index(w)
                        o_cnts[i] += 1
                    except ValueError:
                        o_wrds.append(w)
                        o_cnts.append(1)
            else:
                fp_hldout.write(' ' + sent)
                for w in sent.split('> ')[1].split():
                    try:
                        i = h_wrds.index(w)
                        h_cnts[i] += 1
                    except ValueError:
                        h_wrds.append(w)
                        h_cnts.append(1)
        
        txt = ' '.join(['%s:%s' %(w, o_cnts[i]) for i,w in enumerate(o_wrds)])
        fp_lda_obsvd.write('%d %s\n' %(len(o_wrds), txt))
        txt = ' '.join(['%s:%s' %(w, h_cnts[i]) for i,w in enumerate(h_wrds)])
        fp_lda_hldout.write('%d %s\n' %(len(h_wrds), txt))

        fp_obsvd.write('\n')
        fp_hldout.write('\n')


    fp_hldout.close()
    fp_obsvd.close()
    fp_lda_hldout.close()
    fp_lda_obsvd.close()
    fpin.close()
