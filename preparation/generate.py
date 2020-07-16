import xml.etree.ElementTree as ET

def generate(file_name):
    tree = ET.ElementTree(file=file_name)
    root = tree.getroot()

    sentences = []
    poss = []
    targets = []
    targets_index_start = []
    targets_index_end = []
    lemmas = []

    for doc in root:
        for sent in doc:
            sentence = []
            pos = []
            target = []
            target_index_start = []
            target_index_end = []
            lemma = []
            for token in sent:
                assert token.tag == 'wf' or token.tag == 'instance'
                if token.tag == 'wf':
                    for i in token.text.split(' '):
                        sentence.append(i)
                        pos.append(token.attrib['pos'])
                        target.append('X')
                        lemma.append(token.attrib['lemma'])
                if token.tag == 'instance':
                    target_start = len(sentence)
                    for i in token.text.split(' '):
                        sentence.append(i)
                        pos.append(token.attrib['pos'])
                        target.append(token.attrib['id'])
                        lemma.append(token.attrib['lemma'])
                    target_end = len(sentence)
                    assert ' '.join(sentence[target_start:target_end]) == token.text
                    target_index_start.append(target_start)
                    target_index_end.append(target_end)
            sentences.append(sentence)
            poss.append(pos)
            targets.append(target)
            targets_index_start.append(target_index_start)
            targets_index_end.append(target_index_end)
            lemmas.append(lemma)

    
    gold_keys = []
    with open(file_name[:-len('.data.xml')] + '.gold.key.txt', "r", encoding="utf-8") as m:
        key = m.readline().strip().split()
        while key:
            gold_keys.append(key[1])
            key = m.readline().strip().split()


    output_file = file_name[:-len('.data.xml')] + '.csv'
    with open(output_file, "w", encoding="utf-8") as g:
        g.write('sentence\ttarget_index_start\ttarget_index_end\ttarget_id\ttarget_lemma\ttarget_pos\tsense_key\n')
        num = 0
        for i in range(len(sentences)):
            for j in range(len(targets_index_start[i])):
                sentence = ' '.join(sentences[i])
                target_start = targets_index_start[i][j]
                target_end = targets_index_end[i][j]
                target_id = targets[i][target_start]
                target_lemma = lemmas[i][target_start]
                target_pos = poss[i][target_start]
                sense_key = gold_keys[num]
                num += 1
                g.write('\t'.join((sentence, str(target_start), str(target_end), target_id, target_lemma, target_pos, sense_key)))
                g.write('\n')

if __name__ == "__main__":
    eval_dataset = ['senseval2', 'senseval3', 'semeval2007', 'semeval2013', 'semeval2015', 'ALL']
    # train_dataset = ['SemCor', 'SemCor+OMSTI']
    train_dataset = ['SemCor']

    file_name = []
    for dataset in eval_dataset:
        file_name.append('./Evaluation_Datasets/' + dataset + '/' + dataset + '.data.xml')
    for dataset in train_dataset:
        file_name.append('./Training_Corpora/' + dataset + '/' + dataset.lower() + '.data.xml')

    for file in file_name:
        print(file)
        generate(file)
