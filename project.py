
import pandas as pd
import numpy as np
import os
import re
import requests
import time

def get_book(url):
    response = requests.get(url)
    text_content = response.text
    time.sleep(5)

    first_ind = text_content.find("***")
    second_ind = text_content.find("***", first_ind + 1)
    end_marker = text_content.find("*** END ")

    relevant = text_content[second_ind+3:end_marker]
    text = relevant.replace('\r\n', '\n')

    return text

def tokenize(book_string):
    book_string = re.sub(r'\n{2,}', ' \x03 \x02 ', book_string)

    tokens = re.findall(r'[\w]+|[^\s]', book_string)

    if tokens[0] == '\x03' and tokens[-1] == '\x02':
        return tokens[1:-1]
    elif tokens[0] == '\x03' and tokens[-1] != '\x02':
        return tokens[1:] + ['\x03']
    elif tokens[0] != '\x03' and tokens[-1] == '\x02':
        return ['\x02'] + tokens[:-1]
    else:
        return ['\x02'] + tokens + ['\x03']


class UniformLM(object):


    def __init__(self, tokens):

        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        unique_tokens = tuple(set(tokens))
        x = pd.DataFrame(unique_tokens, columns = ['tokens'])
        x['helper'] = 1
        counts = x.groupby('tokens').count()['helper']
        return counts / len(unique_tokens)
    
    def probability(self, words):
        result = 1
        for i in words:
            if i not in self.mdl.index:
                return 0
            result *= self.mdl[i]
        return result
        
    def sample(self, M):
        sample = np.random.choice(self.mdl.index, size=M, replace=True, p=self.mdl.values)
        return ' '.join(sample)


class UnigramLM(object):
    
    def __init__(self, tokens):

        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        x = pd.DataFrame()
        x['tokens'] = tokens
        x['helper'] = 1
        counts = x.groupby('tokens').count()['helper']
        return counts / len(tokens)


    def probability(self, words):
        result = 1
        for i in words:
            if i not in self.mdl.index:
                return 0
            result *= self.mdl[i]
        return result
        
    def sample(self, M):
        sample = np.random.choice(self.mdl.index, size=M, replace=True, p=self.mdl.values)
        return ' '.join(sample)


class NGramLM(object):
    
    def __init__(self, N, tokens):
        # You don't need to edit the constructor,
        # but you should understand how it works!
        
        self.N = N

        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)

    def create_ngrams(self, tokens):
        n_grams = []
        count = 0
        while (count + self.N) <= len(tokens):
            curr_N = tokens[count:count + self.N]
            n_grams.append(tuple(curr_N))
            count += 1
        return n_grams

        
    def train(self, ngrams):

        unique_ngrams = []
        n1gram_lst = []
        all_n1gram = []
        to_check = {}

        for i in ngrams:
            all_n1gram.append(i[:-1])
            if i not in to_check:
                to_check[i] = 'in'
                unique_ngrams.append(i)
                n1gram_lst.append(i[:-1])

        df = pd.DataFrame()
        df['ngram'] = unique_ngrams
        df['n1gram'] = n1gram_lst

        df2 = pd.DataFrame(pd.Series(all_n1gram), columns = ['n1gram'])
        df2['count'] = 1
        df2 = df2.groupby('n1gram').count()
        df2 = df2.rename(columns={'count': 'denom'})

        df3 = pd.DataFrame(pd.Series(ngrams), columns = ['ngram'])
        df3['count'] = 1
        df3 = df3.groupby('ngram').count()
        df3 = df3.rename(columns={'count': 'numer'})

        merged = df.merge(df2, left_on='n1gram', right_index = True)
        ovr_merge = merged.merge(df3, left_on='ngram', right_index = True)
        ovr_merge['prob'] = ovr_merge['numer'] / ovr_merge['denom']
        ovr_merge = ovr_merge.drop(columns = ['denom', 'numer'])

        return ovr_merge


    
    def probability(self, words):
        words_ngrams = self.create_ngrams(words)
        prob = 1
        for i in words_ngrams:
            if i not in list(self.mdl['ngram']):
                return 0
            value = self.mdl[self.mdl['ngram'] == i]['prob'].squeeze()
            prob *= value
        prob *= self.prev_mdl.probability(words[:self.N - 1])
        return prob

    

    def sample(self, M):
        def helper(lst):
            if len(lst) < self.N - 1:
                previous = self.prev_mdl
                while len(lst) != len(previous.mdl['n1gram'].iloc[0]):
                    previous = previous.prev_mdl
                correct_n1 = previous.mdl[previous.mdl['n1gram'] == tuple(lst)]
                if len(correct_n1) == 0:
                    sample = '\x03'
                else:
                    sample = np.random.choice(correct_n1['ngram'].values, size=1, replace=True, p=correct_n1['prob'].values)
            else:
                correct_n1 = self.mdl[self.mdl['n1gram'] == tuple(lst[(-1 * self.N) + 1:])]
                if len(correct_n1) == 0:
                    sample = '\x03'
                else:
                    sample = np.random.choice(correct_n1['ngram'].values, size=1, replace=True, p=correct_n1['prob'].values)
            return sample[0][-1]
        
        return_list = ["\x02"]

        for i in range(M - 1):
            return_list += [helper(return_list)]
        return_list.append("\x03")
        return " ".join(return_list)
