#!/bin/bash

python train.py  --exp_name unsupMT_enro  --dump_path models/ \
                 --reload_model 'mass_enro_1024.pth,mass_enro_1024.pth'  --data_path 'data/processed/en-ro' \
                 --lgs 'en-ro'  --ae_steps 'en,ro'  --bt_steps 'en-ro-en,ro-en-ro'  --c0 0.01  --T 2000 \
                 --competence_type 'sqrt'  --diff_type 'tfidf_wordtrans'  --diff_file_prefix 'data/processed/en-ro/tfidf_loglen_wordtrans' \
                 --word_shuffle 3  --word_dropout 0.1  --word_blank 0.1  --encoder_only false \
                 --bt_cl true  --ae_js true  --bt_sp true  --bt_sentence true  --bt_word true  --tokens_per_batch 2000  --batch_size 16  --bptt 256 \
                 --emb_dim 1024  --n_layers 6  --n_heads 8  --dropout 0.1  --attention_dropout 0.1  --gelu_activation true \
                 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.998,lr=0.0001  --epoch_size 50000  --eval_bleu true \
                 --stopping_criterion 'valid_ro-en_mt_bleu,10'  --validation_metrics 'valid_en-ro_mt_bleu,valid_ro-en_mt_bleu'
