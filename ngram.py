from datetime import datetime

import csv
import re
import gzip
import nltk

STEMMED_WORDS = nltk.corpus.stopwords.words('english')
SNOWBALL_STEMMER = nltk.stem.SnowballStemmer('english')
NEWLINE_CONSTANT = 'ada770804a0b11e5885dfeff819cdc9f'
unigram_cache = []

def main_program():
    sentence_list = list(open('input/input.txt', 'r'))

    print('Type of sentence_list:')
    print(type(sentence_list))

    get_ngram_for_string(sentence_list, 1, True)
    get_ngram_for_string(sentence_list, 2, True)
    get_ngram_for_string(sentence_list, 3, True)

def list_subtraction(minuend, subtrahend):
    return [item for item in minuend if item not in subtrahend]

def get_unigrams(sentence, get_base_words=False):
    global unigram_cache
    if not unigram_cache:
        tokens = nltk.word_tokenize(sentence)

        if get_base_words:
            # tokens = [SNOWBALL_STEMMER.stem(token) for token in tokens]
            tokens = map(SNOWBALL_STEMMER.stem, tokens)

        unigram_cache = list_subtraction(tokens, STEMMED_WORDS)

    return unigram_cache

def get_bigrams(sentence, get_base_words):
    nltk_bigrams = nltk.bigrams(get_unigrams(sentence, get_base_words))
    return [' '.join(pair) for pair in nltk_bigrams]

def get_trigrams(sentence, get_base_words):
    nltk_trigrams = nltk.trigrams(get_unigrams(sentence, get_base_words))
    return [' '.join(trio) for trio in nltk_trigrams]

def tag_cloud(tokens):
    return nltk.FreqDist(tokens).items()

def tag_cloud_to_file(tag_cloud, filename, sort_input=False):
    if sort_input:
        tag_cloud.sort()

    with open(filename, 'w', newline='') as file_path:
        writer = csv.writer(file_path)
        writer.writerows(tag_cloud)

def join_sentence_list(sentence_list):
    combined_string = (' ' + str(NEWLINE_CONSTANT) + ' ').join(sentence_list)
    combined_string = re.sub(r'([^\s\w]|_)+', '', combined_string)
    combined_string = combined_string.lower()
    return combined_string

def get_ngram_for_string(sentence_list, ntype, get_base_words):
    input_string = join_sentence_list(sentence_list)
    if ntype == 1:
        ngrams = get_unigrams(input_string, get_base_words)
    elif ntype == 2:
        ngrams = get_bigrams(input_string, get_base_words)
    elif ntype == 3:
        ngrams = get_trigrams(input_string, get_base_words)
    else:
        print ('not supported')
        return

    ngram_tagcloud = tag_cloud(ngrams)

    ngram_no_newlines = []
    for (ngram, count) in ngram_tagcloud:
        if NEWLINE_CONSTANT not in ngram:
            ngram_no_newlines.append((ngram, count))

    output_path = 'output/' + str(ntype) + '_ngram_tagcloud.csv'
    tag_cloud_to_file(ngram_no_newlines, output_path)

if __name__ == '__main__':
    startTime = datetime.now()
    main_program()
    print (datetime.now() - startTime)
