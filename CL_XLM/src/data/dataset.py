# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import math
import numpy as np
import torch


logger = getLogger()


class StreamDataset(object):

    def __init__(self, sent, pos, bs, params):
        """
        Prepare batches for data iterator.
        """
        bptt = params.bptt
        self.eos = params.eos_index

        # checks
        assert len(pos) == (sent == self.eos).sum()
        assert len(pos) == (sent[pos[:, 1]] == self.eos).sum()

        n_tokens = len(sent)
        n_batches = math.ceil(n_tokens / (bs * bptt))
        t_size = n_batches * bptt * bs

        buffer = np.zeros(t_size, dtype=sent.dtype) + self.eos
        buffer[t_size - n_tokens:] = sent
        buffer = buffer.reshape((bs, n_batches * bptt)).T
        self.data = np.zeros((n_batches * bptt + 1, bs), dtype=sent.dtype) + self.eos
        self.data[1:] = buffer

        self.bptt = bptt
        self.n_tokens = n_tokens
        self.n_batches = n_batches
        self.n_sentences = len(pos)
        self.lengths = torch.LongTensor(bs).fill_(bptt)

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return self.n_sentences

    def select_data(self, a, b):
        """
        Only select a subset of the dataset.
        """
        if not (0 <= a < b <= self.n_batches):
            logger.warning("Invalid split values: %i %i - %i" % (a, b, self.n_batches))
            return
        assert 0 <= a < b <= self.n_batches
        logger.info("Selecting batches from %i to %i ..." % (a, b))

        # sub-select
        self.data = self.data[a * self.bptt:b * self.bptt]
        self.n_batches = b - a
        self.n_sentences = (self.data == self.eos).sum().item()

    def get_iterator(self, shuffle, subsample=1):
        """
        Return a sentences iterator.
        """
        indexes = (np.random.permutation if shuffle else range)(self.n_batches // subsample)
        for i in indexes:
            a = self.bptt * i
            b = self.bptt * (i + 1)
            yield torch.from_numpy(self.data[a:b].astype(np.int64)), self.lengths


class Dataset(object):

    def __init__(self, sent, pos, params, lang):

        self.lang = lang
        self.params = params
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.batch_size = params.batch_size
        self.tokens_per_batch = params.tokens_per_batch
        self.max_batch_size = params.max_batch_size

        self.step_counter = 0
        self.last_samples_num = self.params.c0
        self.last_batches_num = 0

        self.sent = sent
        self.pos = pos
        self.lengths = self.pos[:, 1] - self.pos[:, 0]

        # check number of sentences
        assert len(self.pos) == (self.sent == self.eos_index).sum()

        # # remove empty sentences
        # self.remove_empty_sentences()
        # load difficulties
        if self.params.diff_type != 'length':
            assert params.diff_file_prefix != ""
            self.origin_diff = self.load_difficulty(params.diff_file_prefix + '.' + self.lang)

        # sanity checks
        self.check()

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return len(self.pos)


    def load_difficulty(self, filename):
        difficulties = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                difficulties.append(float(line))

        return np.array(difficulties)


    def check(self):
        """
        Sanity checks.
        """
        eos = self.eos_index
        assert len(self.pos) == (self.sent[self.pos[:, 1]] == eos).sum()  # check sentences indices
        # assert self.lengths.min() > 0                                     # check empty sentences

    def batch_sentences(self, sentences):
        """
        Take as input a list of n sentences (torch.LongTensor vectors) and return
        a tensor of size (slen, n) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        # sentences = sorted(sentences, key=lambda x: len(x), reverse=True)
        lengths = torch.LongTensor([len(s) + 2 for s in sentences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.pad_index)

        sent[0] = self.eos_index
        for i, s in enumerate(sentences):
            if lengths[i] > 2:  # if sentence not empty
                sent[1:lengths[i] - 1, i].copy_(torch.from_numpy(s.astype(np.int64)))
            sent[lengths[i] - 1, i] = self.eos_index

        return sent, lengths

    def remove_empty_sentences(self):
        """
        Remove empty sentences.
        """
        init_size = len(self.pos)
        indices = np.arange(len(self.pos))
        indices = indices[self.lengths[indices] > 0]
        # remove the empty sentences and its corresponding difficulties
        if self.params.diff_type != 'length':
            self.origin_diff = self.origin_diff[indices]
        self.pos = self.pos[indices]
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        logger.info("Removed %i empty sentences." % (init_size - len(indices)))
        self.check()

    def remove_long_sentences(self, max_len):
        """
        Remove sentences exceeding a certain length.
        """
        assert max_len >= 0
        if max_len == 0:
            return
        init_size = len(self.pos)
        indices = np.arange(len(self.pos))
        indices = indices[self.lengths[indices] <= max_len]
        # remove the long sentences and its corresponding difficulties
        if self.params.diff_type != 'length':
            self.origin_diff = self.origin_diff[indices]
        self.pos = self.pos[indices]
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        logger.info("Removed %i too long sentences." % (init_size - len(indices)))
        self.check()

    def select_data(self, a, b):
        """
        Only select a subset of the dataset.
        """
        assert 0 <= a < b <= len(self.pos)
        logger.info("Selecting sentences from %i to %i ..." % (a, b))

        # sub-select
        self.pos = self.pos[a:b]
        self.lengths = self.pos[:, 1] - self.pos[:, 0]

        # re-index
        min_pos = self.pos.min()
        max_pos = self.pos.max()
        self.pos -= min_pos
        self.sent = self.sent[min_pos:max_pos + 1]

        # sanity checks
        self.check()

    def compute_difficulty(self):
        # compute difficulty uniformly function
        sorted_index = np.argsort(self.origin_diff, kind='mergesort')

        unnorm_difficulty = np.arange(len(sorted_index))
        norm_difficulty = unnorm_difficulty / unnorm_difficulty.max()

        # restore the cdf scores order
        self.diff = np.zeros(sorted_index.shape)
        for i in range(len(norm_difficulty)):
            self.diff[sorted_index[i]] = norm_difficulty[i]

        return self.diff

    def compute_ct_function(self, step_counter):
        ct = min(1, math.sqrt(step_counter * ((1 - self.params.c0 ** 2) / self.params.T) + self.params.c0 ** 2))
        return ct

    def get_continus_batches_iterator(self, batches, return_indices):
        """
        Return a sentences iterator, given the associated sentence batches.
        """
        assert type(return_indices) is bool

        for sentence_ids in batches:
            self.step_counter += 1
            if 0 < self.max_batch_size < len(sentence_ids):
                np.random.shuffle(sentence_ids)
                sentence_ids = sentence_ids[:self.max_batch_size]
            pos = self.pos[sentence_ids]
            sent = [self.sent[a:b] for a, b in pos]
            sent = self.batch_sentences(sent)
            yield (sent, sentence_ids) if return_indices else sent


    def get_iterator(self, iter_name, shuffle, group_by_size=False, n_sentences=-1, seed=None, return_indices=False,
                     params=None, loss_history=None, current_loss=None):
        """
        Return a sentences iterator.
        """
        assert seed is None or shuffle is True and type(seed) is int
        n_sentences = len(self.pos) if n_sentences == -1 else n_sentences
        assert 0 < n_sentences <= len(self.pos)
        assert type(shuffle) is bool and type(group_by_size) is bool
        assert group_by_size is False or shuffle is True

        assert len(self.lengths) == n_sentences
        rng = np.random.RandomState(seed)

        # sentence lengths
        lengths = self.lengths + 2

        if iter_name == 'ae' or (iter_name == 'bt' and params.bt_cl == False):
            # select sentences to iterate over
            indices = rng.permutation(len(self.pos))[:n_sentences]

            # group sentences by lengths
            if group_by_size:
                indices = indices[np.argsort(lengths[indices], kind='mergesort')]

            # create batches - either have a fixed number of sentences, or a similar number of tokens
            if self.tokens_per_batch == -1:
                batches = np.array_split(indices, math.ceil(len(indices) * 1. / self.batch_size))
            else:
                batch_ids = np.cumsum(lengths[indices]) // self.tokens_per_batch
                _, bounds = np.unique(batch_ids, return_index=True)
                batches = [indices[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
                if bounds[-1] < len(indices):
                    batches.append(indices[bounds[-1]:])

            # optionally shuffle batches
            if shuffle:
                rng.shuffle(batches)

            # sanity checks
            assert n_sentences == sum([len(x) for x in batches])
            assert lengths[indices].sum() == sum([lengths[x].sum() for x in batches])

            # return the iterator
            return self.get_continus_batches_iterator(batches, return_indices)

        elif (iter_name == 'bt' and params.bt_cl == True):
            # judge the difficulty type, if the difficulty metric is length, do it directly
            if params.diff_type == 'length':
                sorted_index = np.argsort(self.lengths, kind='mergesort')

                histo = np.histogram(lengths, bins=len(lengths))
                cdf = np.cumsum(histo[0])
                lengths_cdf = cdf / np.amax(cdf)

                # define the difficuty based on the cdf of lengths scores:
                difficulty = np.zeros(lengths.shape)
                for i in range(lengths.shape[0]):
                    difficulty[sorted_index[i]] = lengths_cdf[i]

            else:
                difficulty = self.compute_difficulty()

            sorted_indices = np.argsort(difficulty)

            # compute ct based on loss ratio
            ct = self.compute_ct_function(self.last_batches_num)
            f = open("proper_samples_ende_" + self.params.competence_type +  ".txt", "a", encoding="utf-8")
            if self.last_batches_num == 0:
                indices = sorted_indices[0:int(len(self.pos) * self.params.c0)]
                self.last_samples_num = len(indices)
            else:
                indices = sorted_indices[0:min(int(len(self.pos) * ct), len(self.pos))]
                self.last_samples_num = min(int(len(self.pos) * ct), len(self.pos))

            # group sentences by lengths
            if group_by_size:
                indices = indices[np.argsort(lengths[indices], kind='mergesort')]

            # create batches - either have a fixed number of sentences, or a similar number of tokens
            if self.tokens_per_batch == -1:
                batches = np.array_split(indices, math.ceil(len(indices) * 1. / self.batch_size))
            else:
                batch_ids = np.cumsum(lengths[indices]) // self.tokens_per_batch
                _, bounds = np.unique(batch_ids, return_index=True)
                batches = [indices[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
                if bounds[-1] < len(indices):
                    batches.append(indices[bounds[-1]:])

            self.last_batches_num += len(batches)

            # optionally shuffle batches
            if shuffle:
                rng.shuffle(batches)

            f.write(str(ct) + '\t' + str(len(indices)) + '\t' + str(self.last_batches_num) + '\t'
                    + str(self.step_counter) + '\n')
            f.close()

            # return the iterator
            return self.get_continus_batches_iterator(batches, return_indices)


class ParallelDataset(Dataset):

    def __init__(self, sent1, pos1, sent2, pos2, params):

        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.batch_size = params.batch_size
        self.tokens_per_batch = params.tokens_per_batch
        self.max_batch_size = params.max_batch_size

        self.sent1 = sent1
        self.sent2 = sent2
        self.pos1 = pos1
        self.pos2 = pos2
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]

        # check number of sentences
        assert len(self.pos1) == (self.sent1 == self.eos_index).sum()
        assert len(self.pos2) == (self.sent2 == self.eos_index).sum()

        # remove empty sentences
        self.remove_empty_sentences()

        # sanity checks
        self.check()

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return len(self.pos1)

    def check(self):
        """
        Sanity checks.
        """
        eos = self.eos_index
        assert len(self.pos1) == len(self.pos2) > 0                          # check number of sentences
        assert len(self.pos1) == (self.sent1[self.pos1[:, 1]] == eos).sum()  # check sentences indices
        assert len(self.pos2) == (self.sent2[self.pos2[:, 1]] == eos).sum()  # check sentences indices
        assert eos <= self.sent1.min() < self.sent1.max()                    # check dictionary indices
        assert eos <= self.sent2.min() < self.sent2.max()                    # check dictionary indices
        assert self.lengths1.min() > 0                                       # check empty sentences
        assert self.lengths2.min() > 0                                       # check empty sentences

    def remove_empty_sentences(self):
        """
        Remove empty sentences.
        """
        init_size = len(self.pos1)
        indices = np.arange(len(self.pos1))
        indices = indices[self.lengths1[indices] > 0]
        indices = indices[self.lengths2[indices] > 0]
        self.pos1 = self.pos1[indices]
        self.pos2 = self.pos2[indices]
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        logger.info("Removed %i empty sentences." % (init_size - len(indices)))
        self.check()

    def remove_long_sentences(self, max_len):
        """
        Remove sentences exceeding a certain length.
        """
        assert max_len >= 0
        if max_len == 0:
            return
        init_size = len(self.pos1)
        indices = np.arange(len(self.pos1))
        indices = indices[self.lengths1[indices] <= max_len]
        indices = indices[self.lengths2[indices] <= max_len]
        self.pos1 = self.pos1[indices]
        self.pos2 = self.pos2[indices]
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        logger.info("Removed %i too long sentences." % (init_size - len(indices)))
        self.check()

    def select_data(self, a, b):
        """
        Only select a subset of the dataset.
        """
        assert 0 <= a < b <= len(self.pos1)
        logger.info("Selecting sentences from %i to %i ..." % (a, b))

        # sub-select
        self.pos1 = self.pos1[a:b]
        self.pos2 = self.pos2[a:b]
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]

        # re-index
        min_pos1 = self.pos1.min()
        max_pos1 = self.pos1.max()
        min_pos2 = self.pos2.min()
        max_pos2 = self.pos2.max()
        self.pos1 -= min_pos1
        self.pos2 -= min_pos2
        self.sent1 = self.sent1[min_pos1:max_pos1 + 1]
        self.sent2 = self.sent2[min_pos2:max_pos2 + 1]

        # sanity checks
        self.check()

    def get_batches_iterator(self, batches=None, return_indices=False, c0=None, T=None, lengths=None, difficulty=None,
                             iter_name=None):
        """
        Return a sentences iterator, given the associated sentence batches.
        """
        assert type(return_indices) is bool

        for sentence_ids in batches:
            if 0 < self.max_batch_size < len(sentence_ids):
                np.random.shuffle(sentence_ids)
                sentence_ids = sentence_ids[:self.max_batch_size]
            pos1 = self.pos1[sentence_ids]
            pos2 = self.pos2[sentence_ids]
            sent1 = self.batch_sentences([self.sent1[a:b] for a, b in pos1])
            sent2 = self.batch_sentences([self.sent2[a:b] for a, b in pos2])
            yield (sent1, sent2, sentence_ids) if return_indices else (sent1, sent2)

    def get_iterator(self, iter_name, shuffle, group_by_size=False, n_sentences=-1, seed=None, return_indices=False,
                     params=None, loss_history=None, current_loss=None):
        """
        Return a sentences iterator.
        """
        n_sentences = len(self.pos1) if n_sentences == -1 else n_sentences
        assert 0 < n_sentences <= len(self.pos1)
        assert type(shuffle) is bool and type(group_by_size) is bool

        # sentence lengths
        lengths = self.lengths1 + self.lengths2 + 4

        # select sentences to iterate over
        if shuffle:
            indices = np.random.permutation(len(self.pos1))[:n_sentences]
        else:
            indices = np.arange(n_sentences)

        # group sentences by lengths
        if group_by_size:
            indices = indices[np.argsort(lengths[indices], kind='mergesort')]

        # create batches - either have a fixed number of sentences, or a similar number of tokens
        if self.tokens_per_batch == -1:
            batches = np.array_split(indices, math.ceil(len(indices) * 1. / self.batch_size))
        else:
            batch_ids = np.cumsum(lengths[indices]) // self.tokens_per_batch
            _, bounds = np.unique(batch_ids, return_index=True)
            batches = [indices[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
            if bounds[-1] < len(indices):
                batches.append(indices[bounds[-1]:])

        # optionally shuffle batches
        if shuffle:
            np.random.shuffle(batches)

        # sanity checks
        assert n_sentences == sum([len(x) for x in batches])
        assert lengths[indices].sum() == sum([lengths[x].sum() for x in batches])
        # assert set.union(*[set(x.tolist()) for x in batches]) == set(range(n_sentences))  # slow

        # return the iterator
        return self.get_batches_iterator(batches, return_indices)
