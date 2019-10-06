import pandas as pd
import numpy as np
import argparse
import os
import pickle

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
    
    gold_key_file_name = './Evaluation_Datasets/'+dataset+'/'+dataset+'.gold.key.txt'
    heads=[]
    with open(gold_key_file_name,"r",encoding="utf-8") as f:
        s=f.readline().strip()
        while s:
            heads.append(s.split()[0])
            s = f.readline().strip()
    

    with open('./wordnet/index2sense.pkl', 'rb') as h:
        index2sense = pickle.load(h)


    labels=[]
    with open(input_file_name, "r", encoding="utf-8") as g:
        s=g.readline().strip()
        while s:
            labels.append(index2sense[int(s.split()[0])])
            s=g.readline().strip()
 
    with open(os.path.join(output_dir, "final_result_"+dataset+'.txt'),"w",encoding="utf-8") as m:
        for i in range(len(heads)):
            m.write(heads[i]+" "+labels[i]+'\n')

if __name__ == "__main__":
    main()