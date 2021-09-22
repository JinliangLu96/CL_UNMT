#!/bin/bash

python train.py  --exp_name unsupMT_ende  --dump_path models/ \
                 --reload_model 'mlm_ende_1024.pth,mlm_ende_1024.pth' \
                 --data_path 'data/processed/de-en-1m/'  --lgs 'en-de'  --ae_steps 'en,de'  --bt_steps 'en-de-en,de-en-de' \
                 --word_shuffle 3  --word_dropout 0.1  --word_blank 0.1  --lambda_ae '0:1,100000:0.1,300000:0' --encoder_only false \
                 --bt_cl false  --ae_js true  --bt_sp true  --bt_sentence true  --bt_word true  --tokens_per_batch 2000  --batch_size 16  --bptt 256 \
                 --emb_dim 1024  --n_layers 6  --n_heads 8  --dropout 0.1  --attention_dropout 0.1  --gelu_activation true \
                 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.998,lr=0.0001,warmup_updates=10000,weight_decay=0.01,eps=0.000001 \
                 --epoch_size 50000  --eval_bleu true  --stopping_criterion 'valid_de-en_mt_bleu,10'  --validation_metrics 'valid_en-de_mt_bleu,valid_de-en_mt_bleu'
