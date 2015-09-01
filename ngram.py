import csv
import re
import gzip
import nltk

STOPWORDS = nltk.corpus.stopwords.words('english')
SNOWBALL_STEMMER = nltk.stem.SnowballStemmer('english')
NEWLINE_CONSTANT = 'ada770804a0b11e5885dfeff819cdc9f'
unigram_cache = []

def main_program():
    with open('input/input.txt', 'r') as f:
        sentence_list = f.readlines()

    input_string = join_sentence_list(sentence_list)

    get_ngram_for_string(input_string, 1, True)
    get_ngram_for_string(input_string, 2, True)
    get_ngram_for_string(input_string, 3, True)

def get_unigrams(sentence, get_base_words=False):
    global unigram_cache
    if not unigram_cache:
        tokens = nltk.word_tokenize(sentence)

        if get_base_words:
            unigram_cache = [SNOWBALL_STEMMER.stem(token) for token in tokens]

        unigram_cache = tokens

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
    stopwords_pattern = re.compile(r'\b(' + r'|'.join(STOPWORDS) + r')\b\s*')
    whitespace_pattern = re.compile(r'([^\s\w]|_)+')

    combined_string = (' ' + str(NEWLINE_CONSTANT) + ' ').join(sentence_list)
    combined_string = combined_string.lower()
    combined_string = stopwords_pattern.sub(' ', combined_string)
    combined_string = whitespace_pattern.sub('', combined_string)

    return combined_string

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
        if NEWLINE_CONSTANT not in ngram:
            ngram_no_newlines.append((ngram, count))

    output_path = 'output/' + str(ntype) + '_ngram_tagcloud.csv'
    tag_cloud_to_file(ngram_no_newlines, output_path)

if __name__ == '__main__':
    main_program()
