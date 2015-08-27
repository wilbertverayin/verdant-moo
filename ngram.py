import csv
import re
import gzip
import nltk

snowball_stemmer = nltk.stem.SnowballStemmer('english')

NEWLINE_CONSTANT = 'ada770804a0b11e5885dfeff819cdc9f'

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
    tokens = nltk.word_tokenize(sentence)

    if get_base_words:
        tokens = [snowball_stemmer.stem(token) for token in tokens]

    stopwords = nltk.corpus.stopwords.words('english')

    return list_subtraction(tokens, stopwords)

def get_bigrams(sentence, get_base_words):
    nltk_bigrams = nltk.bigrams(get_unigrams(sentence, get_base_words))
    return [' '.join(pair) for pair in nltk_bigrams]

def get_trigrams(sentence, get_base_words):
    nltk_trigrams = nltk.trigrams(get_unigrams(sentence, get_base_words))
    return [' '.join(trio) for trio in nltk_trigrams]

def tag_cloud(tokens):
    return nltk.FreqDist(tokens).items()

def tag_cloud_to_file(tag_cloud, filename):
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
        if not NEWLINE_CONSTANT in ngram:
            ngram_no_newlines.append((ngram, count))

    tag_cloud_to_file(
        ngram_no_newlines,
        'output/' + str(ntype) + '_ngram_tagcloud.csv'
    )

if main_program:
    main_program()
