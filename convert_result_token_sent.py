import pandas as pd
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default=None,
                        type=str,
                        required=True,
                        choices=['senseval2', 'senseval3', 'semeval2007', 'semeval2013', 'semeval2015', 'ALL'],
                        help="Dataset name")
    parser.add_argument("--input_file",
                        default=None,
                        type=str,
                        required=True,
                        help="Input file of results")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="Output dir of final results")
    args = parser.parse_args()


    dataset = args.dataset
    input_file_name = args.input_file
    output_dir = args.output_dir
    train_file_name = './Evaluation_Datasets/'+dataset+'/'+dataset+'.csv'
    train_data = pd.read_csv(train_file_name,sep="\t",na_filter=False).values
    words_train = []
    for i in range(len(train_data)):
        words_train.append(train_data[i][4]) # get lemmas

    test_file_name = './Evaluation_Datasets/'+dataset+'/'+dataset+'_test_sent_cls.csv'
    test_data = pd.read_csv(test_file_name,sep="\t",na_filter=False).values

    seg = [0]
    for i in range(1,len(test_data)):
        if test_data[i][0] != test_data[i-1][0]:
            seg.append(i)
            

    results=[]
    num=0
    with open(input_file_name, "r", encoding="utf-8") as f:
        s=f.readline().strip()
        while s:
            q=float(s.split()[-1])
            results.append((q,test_data[num][-1]))
            num+=1
            s = f.readline().strip()


    with open(os.path.join(output_dir, "final_result_"+dataset+'.txt'),"w",encoding="utf-8") as f:
        for i in range(len(seg)):
            f.write(test_data[seg[i]][0]+" ")
            if i!=len(seg)-1:
                result=results[seg[i]:seg[i+1]]
            else:
                result=results[seg[i]:-1]
            result.sort(key=lambda x:x[0],reverse=True)
            f.write(result[0][1]+"\n")


if __name__ == "__main__":
    main()