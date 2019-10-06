import pandas as pd


def generate_auxiliary(gold_key_file_name, train_file_name, train_file_final_name):

    sense_data = pd.read_csv("./wordnet/index.sense.gloss",sep="\t",header=None).values
    print(len(sense_data))
    print(sense_data[1])


    d = dict()
    for i in range(len(sense_data)):
        s = sense_data[i][0]
        pos = s.find("%")
        try:
            d[s[:pos + 2]].append((sense_data[i][0],sense_data[i][-1]))
        except:
            d[s[:pos + 2]]=[(sense_data[i][0], sense_data[i][-1])]

    print(len(d))
    print(len(d["happy%3"]))
    print(d["happy%3"])
    print(len(d["happy%5"]))
    print(d["happy%5"])
    print(len(d["hard%3"]))
    print(d["hard%3"])



    train_data = pd.read_csv(train_file_name,sep="\t",na_filter=False).values
    print(len(train_data))
    print(train_data[0])

    gold_keys=[]
    with open(gold_key_file_name,"r",encoding="utf-8") as f:
        s=f.readline().strip()
        while s:
            tmp = s.split()[1:]
            gold_keys.append(tmp)
            s=f.readline().strip()
    print(len(gold_keys))
    print(gold_keys[6])

    with open(train_file_final_name,"w",encoding="utf-8") as f:
        f.write('target_id\tlabel\tsentence\tgloss\tsense_key\n')
        for i in range(len(train_data)):
            assert train_data[i][-2]=="NOUN" or train_data[i][-2]=="VERB" or train_data[i][-2]=="ADJ" or train_data[i][-2]=="ADV"
            orig_sentence = train_data[i][0].split(' ')
            start_id = int(train_data[i][1])
            end_id = int(train_data[i][2])
            sentence = []
            for w in range(len(orig_sentence)):
                if w == start_id or w == end_id:
                    sentence.append('"')
                sentence.append(orig_sentence[w])
            sentence = ' '.join(sentence)
            orig_word = ' '.join(orig_sentence[start_id:end_id])
            
            for category in ["%1", "%2", "%3", "%4", "%5"]:
                word = train_data[i][-3]
                query = word+category
                try:
                    sents = d[query]
                    gold_key_exist = 0
                    for j in range(len(sents)):
                        if sents[j][0] in gold_keys[i]:
                            f.write(train_data[i][3]+"\t"+"1")
                            gold_key_exist = 1
                        else:
                            f.write(train_data[i][3]+"\t"+"0")
                        f.write("\t"+train_data[i][0]+"\t"+sents[j][1]+"\t"+sents[j][0]+"\n")
                    assert gold_key_exist == 1
                except:
                    pass



if __name__ == "__main__":
    eval_dataset = ['senseval2', 'senseval3', 'semeval2007', 'semeval2013', 'semeval2015', 'ALL']
    train_dataset = ['SemCor']

    file_path = []
    for dataset in eval_dataset:
        file_path.append('./Evaluation_Datasets/' + dataset + '/' + dataset)
    for dataset in train_dataset:
        file_path.append('./Training_Corpora/' + dataset + '/' + dataset.lower())

    for file_name in file_path:
        gold_key_file_name = file_name + '.gold.key.txt'
        train_file_name = file_name + '.csv'
        if file_name == './Training_Corpora/SemCor/semcor':
            train_file_final_name = file_name + '_train_sent_cls.csv'
        else:
            train_file_final_name = file_name + '_test_sent_cls.csv'

        print(gold_key_file_name)
        print(train_file_name)
        print(train_file_final_name)

        generate_auxiliary(gold_key_file_name, train_file_name, train_file_final_name)