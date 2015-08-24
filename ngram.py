import csv
import re
import gzip
import nltk

sno = nltk.stem.SnowballStemmer('english')

NEWLINE_CONSTANT = ' ada770804a0b11e5885dfeff819cdc9f '

def main_program():
    with open('input.txt') as f:
        content = f.read()

    content = content.replace('\n', NEWLINE_CONSTANT)
    input_string = re.sub(r'([^\s\w]|_)+', '', content)
    input_string = input_string.lower()

    get_ngram_for_string(input_string, 1, True)
    get_ngram_for_string(input_string, 2, True)
    get_ngram_for_string(input_string, 3, True)

def list_subtraction(minuend, subtrahend):
    return [item for item in minuend if item not in subtrahend]

def get_unigrams(sentence, get_base_words):
    tokens = nltk.word_tokenize(sentence)

    if get_base_words:
        tokens = [sno.stem(token) for token in tokens]

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
    with open(filename, 'w', newline='') as file_path:
        writer = csv.writer(file_path)
        writer.writerows(tag_cloud)

def get_ngram_for_string(input_string, ntype, get_base_words):
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
        if ngram.find(NEWLINE_CONSTANT.strip()) == -1:
            ngram_no_newlines.append((ngram, count))

    tag_cloud_to_file(ngram_no_newlines, str(ntype) + '_ngram_tagcloud.csv')
    return

main_program()
