# -*- coding: utf-8 -*-
import re
import string
import io
from collections import Counter
import codecs



# Source: https://github.com/commonsense/metanl/blob/master/metanl/token_utils.py
def untokenize(words):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" ` ", " '")
    return step5.strip()


def tokenize(text):
    return re.findall(r"\s+", text)


class Spell(object):
    def __init__(self, path): # 
        self.dictionary = Counter(self.words(io.open(path, 'r',encoding='utf-8').read()))
        print("dictionary" + str(self.dictionary))

    def words(self, text):
        text = text.replace('\n', ' ')
        hangul = re.compile('[^ a-z가-힣+]')
        result = hangul.sub('', text)
        result = list(result)
        return result

    def P(self, word, N=None):
        "Probability of `word`."
        if N is None:
            N = sum(self.dictionary.values())
        return self.dictionary[word] / N

    def correction(self, word):
        "Most probable spelling correction for word."
        return max(self.candidates(word), key=self.P)

    def candidates(self, word):
        "Generate possible spelling corrections for word."
        return (self.known([word]) or self.known([word]) or [word] or [word])

    def known(self, words):
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.dictionary)



    # Correct words
    def corrections(self, words):
        return [self.correction(word) for word in words]

    # Correct sentence
    def sentence(self, sentence):
        return untokenize(self.corrections(tokenize(sentence)))