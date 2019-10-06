from collections import OrderedDict
import pickle

sense2index = OrderedDict() 
index2sense = OrderedDict()
lemma2sense_dict = OrderedDict()
lemma2index_dict = OrderedDict()

with open("./wordnet/index.sense", "r", encoding='utf-8') as f:
    s = f.readline().strip().split()
    while s:
        sense2index[s[0]] = len(sense2index)
        index2sense[len(index2sense)] = s[0]
        pos = s[0].find("%")
        lemma = s[0][:pos]
        try:
            lemma2sense_dict[lemma].append(s[0])
        except:
            lemma2sense_dict[lemma] = [s[0]]
        
        try:
            lemma2index_dict[lemma].append(sense2index[s[0]])
        except:
            lemma2index_dict[lemma] = [sense2index[s[0]]]

        s = f.readline().strip().split()

with open ('./wordnet/sense2index.pkl', 'wb') as g, open('./wordnet/index2sense.pkl', 'wb') as h, \
open('./wordnet/lemma2sense_dict.pkl', 'wb') as j, open('./wordnet/lemma2index_dict.pkl', 'wb') as k:
    pickle.dump(sense2index, g)
    pickle.dump(index2sense, h)
    pickle.dump(lemma2sense_dict, j)
    pickle.dump(lemma2index_dict, k)

     


