# coding = utf-8
import sys

import math
import numpy as np
from collections import defaultdict

def save_diff(outfile, scores):
    with open(outfile, 'w', encoding='utf-8') as f:
        for s in scores:
            f.write(str(s) + '\n')


def load_dict(filename):
    src2tgt = {}
    src2scores = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split('\t')
            src_word = words[0]
            tgt_word = words[1]
            score = float(words[2])
            if src_word not in src2tgt.keys():
                src2tgt[src_word] = tgt_word
                src2scores[src_word] = score
            else:
                print(src_word)

    return src2tgt, src2scores


def load_sentences(filename):
    sentences = []
    lengths = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            lengths.append(len(line.strip().split()))
            line = line.strip().replace('@@ ', '').split()
            line = [w.lower() for w in line]
            sentences.append(line)

    return sentences, lengths


def stat_docs(corpus):
    w2docs = defaultdict(set)
    for i, sen in enumerate(corpus):
        for w in sen:
            # record documents
            w2docs[w].add(i)

    return w2docs


def computeIDF(corpus):
    w2docs = stat_docs(corpus)
    total_doc_num = len(corpus)
    w2idf = {}
    for w in w2docs.keys():
        w2idf[w] = math.log(float(total_doc_num / (len(w2docs[w]) + 1)))

    return w2idf

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

def stat_length(lengths):
    lengths = np.array(lengths)
    return lengths

def replace_and_compute_score(sentences, diction, src2scores, w2idf, normlength):
    difficulty_scores = []
    # process each sentence
    for s_index, s in enumerate(sentences):
        if s_index % 100000 == 0:
            print('{} sentences - difficulties have been computed.'.format(s_index))
        s_penalty = math.log10(normlength[s_index])
        w2counter = defaultdict(int)
        for w in s:
            w2counter[w] += 1
        # compute tf score
        w2tf_idf = defaultdict(float)
        for w in s:
            w2tf_idf[w] = float(w2counter[w] / sum(w2counter.values())) * w2idf[w]

        # compute difficulty score
        current_score = 0.0
        real_sum_tfidf = 0.0
        for w in s:
            if w in diction.keys():
                score_ws = src2scores[w]
                print(w + ' ' + str(1.0 - score_ws))
                current_score += (1.0 - score_ws) * w2tf_idf[w]
                real_sum_tfidf += w2tf_idf[w]
            elif is_number(w):
                continue
            else:
                current_score += 1.0 * w2tf_idf[w]
                real_sum_tfidf += w2tf_idf[w]
        # print(s_penalty)
        current_score = float(current_score / real_sum_tfidf) * s_penalty
        # print(current_score)
        # current_score = float(current_score / real_sum_tfidf)
        # current_score = 1.0 - current_score
        difficulty_scores.append(current_score)

    return difficulty_scores

if __name__ == "__main__":
    dictionary, src2scores = load_dict(sys.argv[1])
    print("The dictionary has been loaded.")
    sentences, lengths = load_sentences(sys.argv[2])
    print('sentences have been loaded.')
    print("The sentences has been loaded.")
    w2idf = computeIDF(sentences)
    print('tf-idf has been loaded.')
    norm_length = stat_length(lengths)
    difficulty = replace_and_compute_score(sentences, dictionary, src2scores, w2idf, norm_length)

    save_diff(sys.argv[3], difficulty)