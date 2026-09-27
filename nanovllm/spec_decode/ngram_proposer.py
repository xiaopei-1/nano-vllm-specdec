from typing import Optional

class NgramProposer:
    def __init__(self):
        self.min_ngram = 2
        self.max_ngram = 8
        self.k = 4
        self.max_model_len = 8192

    def propose(self,
                 tokens:list[int],
                 min_ngram:Optional[int] = None,
                 max_ngram:Optional[int] = None,
                 max_model_len:Optional[int] = None,
                 k:Optional[int] = None)->list[int]:
        min_ngram = self.min_ngram if min_ngram is None else min_ngram
        max_ngram = self.max_ngram if max_ngram is None else max_ngram
        max_model_len = self.max_model_len if max_model_len is None else max_model_len
        k = self.k if k is None else k

        num_tokens = len(tokens)
        if num_tokens < min_ngram:
            return []
        k = min(k,max_model_len - num_tokens)
        if k <= 0:
            return []

        reversed_tokens = tokens[::-1]
        lps = [0] * num_tokens
        max_ngram_prefix = 0
        pos = 0
        for i in range(1,num_tokens):
            prev_lps = lps[i-1]
            while prev_lps >= max_ngram:
                prev_lps = lps[prev_lps-1]
            while prev_lps > 0 and reversed_tokens[i] != reversed_tokens[prev_lps]:
                prev_lps = lps[prev_lps-1]
            if reversed_tokens[i] == reversed_tokens[prev_lps]:
                lps[i] = prev_lps + 1
            if lps[i] > max_ngram_prefix:
                max_ngram_prefix = lps[i]
                pos = i
        if max_ngram_prefix < min_ngram:
            return []
        start_pos = num_tokens - pos + max_ngram_prefix - 1
        k = min(k,num_tokens - start_pos)
        return tokens[start_pos:start_pos + k]


