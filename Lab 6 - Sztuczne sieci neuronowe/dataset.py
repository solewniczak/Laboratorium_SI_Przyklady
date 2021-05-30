from collections import Counter

import sklearn.model_selection
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

data_set = []
classes = ['spam', 'ham']


tokenizer = RegexpTokenizer(r'\w+')
snowball = SnowballStemmer(language='english')
# Source: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
with open('SMSSpamCollection', 'r', encoding='utf-8') as file:
    for line in file:
        label, msg = line.rstrip().split('\t', 2)
        tokenized_msg = tokenizer.tokenize(msg)
        stemmed_msg = [snowball.stem(token) for token in tokenized_msg]
        data_set.append((label, stemmed_msg))


train_set, test_set = sklearn.model_selection.train_test_split(data_set, train_size=0.8)

vocabulary = Counter()
for label, msg in train_set:
    vocabulary.update(msg)
vocabulary = list(vocabulary)  # Vocabulary must be ordered

print(f'Train len: {len(train_set)}. Test len: {len(test_set)}. Vocabulary len: {len(vocabulary)}')