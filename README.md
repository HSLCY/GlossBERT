# GlossBERT

Codes and corpora for paper "[GlossBERT: BERT for Word Sense Disambiguation with Gloss Knowledge](https://arxiv.org/pdf/1908.07245)" (EMNLP 2019)

## Dependencies

* pytorch: 1.0.0
* python: 3.7.1
* tensorflow: 1.13.1 (only needed for converting BERT-tensorflow-model to pytorch-model)
* numpy: 1.15.4

## Step 1: Preparation

### Datasets and Vocabulary

We generate datasets for GlossBERT based on the evaluation framework of [Raganato et al. ](<http://lcl.uniroma1.it/wsdeval/>) and [WordNet 3.0](https://wordnet.princeton.edu/). 

Run following commands to prepare datasets for tasks and extract vocabulary information from `./wordnet/index.sense` (if you only need the processed datasets, download [here](https://drive.google.com/file/d/1EaAXzVQRI29c3pO8BcKxzrBNgWA5pktR/view?usp=sharing)):

```
bash preparation.sh
```

Then for each dataset, there are 6 files in the directory. Take Semcor as an example:

```
./Training_Corpora/SemCor:
    semcor.csv
    semcor.data.xml
    semcor.gold.key.txt
    semcor_train_sent_cls.csv
    semcor_train_sent_cls_ws.csv
    semcor_train_token_cls.csv
```

- `semcor.data.xml` and `semcor.gold.key.txt` come from the evaluation framework of  [Raganato et al. ](<http://lcl.uniroma1.it/wsdeval/>) 
- `semcor.csv` is generated from `semcor.data.xml` by us, which is used to generate other files and is the dataset for `exp-BERT(Token-CLS)`.
- `semcor_train_sent_cls.csv`, `semcor_train_sent_cls_ws.csv` and `semcor_train_token_cls.csv` are datasets for `exp-GlossBERT(Sent-CLS)`, `exp-GlossBERT(Sent-CLS-WS)` and `exp-GlossBERT(Token-CLS)` respectively.

Besides, four `.pkl` files have been generated in directory:`./wordnet/` , which are need in `exp-BERT(Token-CLS)`.



### BERT-pytorch-model

Download [BERT-Base-uncased model](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) and then run following commands to convert a tensorflow checkpoint to a pytorch model:

```
python convert_tf_checkpoint_to_pytorch.py \
--tf_checkpoint_path bert-model/uncased_L-12_H-768_A-12/bert_model.ckpt \
--bert_config_file bert-model/uncased_L-12_H-768_A-12/bert_config.json \
--pytorch_dump_path bert-model/uncased_L-12_H-768_A-12/pytorch_model.bin
```



## Step 2: Train

For example, for `exp-GlossBERT(Sent-CLS-WS)`:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_classifier_WSD_sent.py \
--task_name WSD \
--train_data_dir ./Training_Corpora/SemCor/semcor_train_sent_cls_ws.csv \
--eval_data_dir ./Evaluation_Datasets/semeval2007/semeval2007_test_sent_cls_ws.csv \
--output_dir results/sent_cls_ws/1314 \
--bert_model ./bert-model/uncased_L-12_H-768_A-12/ \
--do_train \
--do_eval \
--do_lower_case \
--max_seq_length 512 \
--train_batch_size 64 \
--eval_batch_size 128 \
--learning_rate 2e-5 \
--num_train_epochs 6.0 \
--seed 1314
```

See more examples for other experiments in `commands.txt`.



## Step 3: Test

For example, for `exp-GlossBERT(Sent-CLS-WS)`:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_classifier_WSD_sent.py \
--task_name WSD \
--eval_data_dir ./Evaluation_Datasets/senseval3/senseval3_test_sent_cls_ws.csv \
--output_dir results/sent_cls_ws/1314/4 \
--bert_model results/sent_cls_ws/1314/4 \
--do_test \
--do_lower_case \
--max_seq_length 512 \
--train_batch_size 64 \
--eval_batch_size 128 \
--learning_rate 2e-5 \
--num_train_epochs 6.0 \
--seed 1314
```

See more examples for other experiments in `commands.txt`.



## Step 4: Evaluation

Refer to `./Evaluation_Datasets/README` provided by  [Raganato et al. ](<http://lcl.uniroma1.it/wsdeval/>) .

First, you need to convert the output file to make sure its format is the same as the gold key file. You can use code like:

```
# GlossBERT_sent_cls or GlossBERT_sent_cls_ws or GlossBERT_token_cls
python convert_result_token_sent.py \
--dataset semeval2007 \
--input_file ./results/results.txt \
--output_dir ./results/  

# BERT_baseline
python convert_result_baseline.py \
--dataset semeval2007 \
--input_file ./results/results.txt \
--output_dir ./results/
```

Then, you can use the Scorer provided by  [Raganato et al. ](<http://lcl.uniroma1.it/wsdeval/>) to do evaluation.

Example of usage:

```
$ javac Scorer.java
$ java Scorer ./Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt ./results/final_result_semeval2007.txt
```


