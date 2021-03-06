#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
# Modified to support pytorch Tensors

import Levenshtein as Lev
import torch
from six.moves import xrange


class Decoder(object):
    """
    Basic decoder class from which all other decoders inherit. Implements several
    helper functions. Subclasses should implement the decode() method.

    Arguments:
        labels (string): mapping from integers to characters.
        blank_index (int, optional): index for the blank '_' character. Defaults to 0.
        space_index (int, optional): index for the space ' ' character. Defaults to 28.
    """

    def __init__(self, labels, blank_index=0, space_index=28):
        # e.g. labels = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ#"
        self.labels = labels
        self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])
        self.blank_index = blank_index
        self.space_index = space_index

    def convert_to_strings(self, sequences, sizes=None):
        """Given a list of numeric sequences, returns the corresponding strings"""
        strings = []
        for x in xrange(len(sequences)):
            seq_len = sizes[x] if sizes is not None else len(sequences[x])
            string = self._convert_to_string(sequences[x], seq_len)
            strings.append(string)
        return strings

    def _convert_to_string(self, sequence, sizes):
        return ''.join([self.int_to_char[sequence[i].item()] for i in range(sizes)])

    def process_strings(self, sequences, remove_repetitions=False):
        """
        Given a list of strings, removes blanks and replace space character with space.
        Option to remove repetitions (e.g. 'abbca' -> 'abca').

        Arguments:
            sequences: list of 1-d array of integers
            remove_repetitions (boolean, optional): If true, repeating characters
                are removed. Defaults to False.
        """
        processed_strings = []
        for sequence in sequences:
            string = self.process_string(remove_repetitions, sequence).strip()
            processed_strings.append(string)
        return processed_strings

    def process_string(self, remove_repetitions, sequence):
        string = ''
        for i, char in enumerate(sequence):
            if char != self.int_to_char[self.blank_index]:
                # if this char is a repetition and remove_repetitions=true,
                # skip.
                if remove_repetitions and i != 0 and char == sequence[i - 1]:
                    pass
                elif char == self.labels[self.space_index]:
                    string += ' '
                else:
                    string = string + char
        return string

    def wer(self, s1, s2):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """

        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        return Lev.distance(''.join(w1), ''.join(w2))

    def cer(self, s1, s2):
        """
        Computes the Character Error Rate, defined as the edit distance.

        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
        return Lev.distance(s1, s2)

    def decode(self, probs, sizes=None):
        """
        Given a matrix of character probabilities, returns the decoder's
        best guess of the transcription

        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            string: sequence of the model's best guess for the transcription

        """
        raise NotImplementedError


class BeamCTCDecoder(Decoder):
    def __init__(self, labels, beam_width=20, top_paths=1, blank_index=0, space_index=28, lm_path=None, trie_path=None,
                 lm_alpha=None, lm_beta1=None, lm_beta2=None):
        super(BeamCTCDecoder, self).__init__(labels, blank_index=blank_index, space_index=space_index)
        self._beam_width = beam_width
        self._top_n = top_paths

        try:
            from pytorch_ctc import CTCBeamDecoder, Scorer, KenLMScorer
        except ImportError:
            raise ImportError("BeamCTCDecoder requires pytorch_ctc package.")
        if lm_path is not None:
            scorer = KenLMScorer(labels, lm_path, trie_path)
            scorer.set_lm_weight(lm_alpha)
            scorer.set_word_weight(lm_beta1)
            scorer.set_valid_word_weight(lm_beta2)
        else:
            scorer = Scorer()
        self._decoder = CTCBeamDecoder(scorer, labels, top_paths=top_paths, beam_width=beam_width,
                                       blank_index=blank_index, space_index=space_index, merge_repeated=False)

    def decode(self, probs, sizes=None):
        sizes = sizes.cpu() if sizes is not None else None
        out, conf, seq_len = self._decoder.decode(probs.cpu(), sizes)

        # TODO: support returning multiple paths
        strings = self.convert_to_strings(out[0], sizes=seq_len[0])
        return self.process_strings(strings)


class GreedyDecoder(Decoder):
    def decode(self, probs, sizes=None):
        """
        Returns the argmax decoding given the probability matrix. Removes
        repeated elements in the sequence, as well as blanks.

        Arguments:
            probs: Tensor of character probabilities from the network. Expected shape of seq_length x batch x output_dim
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            strings: sequences of the model's best guess for the transcription on inputs
        """
        _, max_probs = torch.max(probs.transpose(0, 1), 2)
        strings = self.convert_to_strings(max_probs.view(max_probs.size(0), max_probs.size(1)), sizes)
        #print(strings)
        return self.process_strings(strings, remove_repetitions=True)
