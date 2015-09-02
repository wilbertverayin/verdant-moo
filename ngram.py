import csv
import re
import gzip
import nltk
import dill

from multiprocessing import Pool
from functools import reduce
from datetime import datetime

STOPWORDS = nltk.corpus.stopwords.words('english')
SNOWBALL_STEMMER = nltk.stem.SnowballStemmer('english')
NEWLINE_CONSTANT = 'ada770804a0b11e5885dfeff819cdc9f'
unigram_cache = []

startTime = None
currentTime = None
SHOW_LOGS = True

def chunkify(input_list, size_per_chunk):
    return [input_list[i:i+size_per_chunk] for i in range(0, len(input_list), size_per_chunk)]

def main_program():
    with open('input/input.txt', 'r') as f:
        sentence_list = f.readlines()

    log('joining sentence lists...')

    input_string = join_sentence_list(sentence_list)

    get_ngram_for_string(input_string, 1, True)
    get_ngram_for_string(input_string, 2, True)
    get_ngram_for_string(input_string, 3, True)

def log(log_string):
    global currentTime
    if SHOW_LOGS:
        print(datetime.now() - startTime)
        currentTime = datetime.now()
        print(log_string, '\n')

def join_lists(prefix_list, suffix_list):
    return prefix_list + suffix_list

def run_dill_encoded(what):
    func, args = dill.loads(what)
    return func(*args)

def apply_async(pool, func, args):
    return pool.apply_async(run_dill_encoded, (dill.dumps((func, args)),))

def stem_words(stemmer, tokens):
    return [stemmer(token) for token in tokens]

def stem_word(stemmer, token):
    return stemmer(token)

def map_stem_words(stemmer, tokens):
    return list(map(stemmer, tokens))

def get_unigrams(sentence, get_base_words=False):
    global unigram_cache
    if not unigram_cache:
        log('caching unigrams...')
        log('tokenizing...')
        tokens = nltk.word_tokenize(sentence)

        if get_base_words:
            log('stemming...')
            # unigram_cache = [SNOWBALL_STEMMER.stem(token) for token in tokens]

            token_chunks = chunkify(tokens, 100000)
            pool = Pool()
            jobs = []

            for chunk in token_chunks:
                job = apply_async(pool, map_stem_words, (SNOWBALL_STEMMER.stem,chunk))
                # job = apply_async(pool, stem_words, (SNOWBALL_STEMMER.stem,chunk))
                jobs.append(job)

            for job in jobs:
                unigram_cache.append(job.get())

            unigram_cache = [item for sublist in unigram_cache for item in sublist]
            # unigram_cache = reduce(join_lists, unigram_cache)

        else:
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

def tag_cloud_to_file(tag_cloud, filename, sort_input=True):
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
    startTime = datetime.now()
    currentTime = datetime.now()

    print('started ngram.py at')
    print(startTime)

    main_program()

    endTime = datetime.now()
    print('\nended at ngram.py at')
    print(endTime)

    print('\nRuntime:')
    print(endTime - startTime)
