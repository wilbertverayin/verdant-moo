import csv
import re
import gzip
import nltk
import dill
import os

from multiprocessing import Pool, cpu_count
from datetime import datetime
from queue import Queue
from threading import Thread

DIRNAME = os.path.dirname(os.path.realpath(__file__))
SNOWBALL_STEMMER = nltk.stem.SnowballStemmer('english').stem
NEWLINE_CONSTANT = 'ada770804a0b11e5885dfeff819cdc9f'

unigram_cache = []
startTime = None
SHOW_LOGS = True

def main_program():
    with open(DIRNAME + '/input/input.txt', 'r') as f:
        sentence_list = f.readlines()

    log('joining sentence lists...')

    input_string = join_sentence_list(sentence_list)

    get_ngram_for_string(input_string, 1, True)
    get_ngram_for_string(input_string, 2, True)
    get_ngram_for_string(input_string, 3, True)

def log(log_string):
    if SHOW_LOGS:
        print(datetime.now() - startTime)
        print(log_string, '\n')

def chunkify(input_list, size_per_chunk):
    return [input_list[i:i+size_per_chunk] for i in range(0, len(input_list), size_per_chunk)]

def run_dill_encoded(what):
    func, args = dill.loads(what)
    return func(*args)

def apply_async(pool, func, args):
    return pool.apply_async(run_dill_encoded, (dill.dumps((func, args)),))

def stem_words(stemmer, tokens):
    return [stemmer(token) for token in tokens]

def stem_word(stemmer, token):
    return stemmer(token)

def stem_tokens(tokens):
    stemmer = SNOWBALL_STEMMER
    return [stemmer(token) for token in tokens]

def multiprocess_stem_tokens(tokens):
    stemmer = SNOWBALL_STEMMER
    process_count = cpu_count()
    chunksize = int(len(tokens) / process_count)
    token_chunks = chunkify(tokens, chunksize)

    pool = Pool()
    jobs = []
    result = []

    for chunk in token_chunks:
        job = apply_async(pool, stem_words, (stemmer, chunk))
        jobs.append(job)

    for job in jobs:
        result.append(job.get())

    return [item for sublist in result for item in sublist]

def do_stuff(q, output_queue):
    while True:
        tokens = q.get()
        output_queue.put([SNOWBALL_STEMMER(token) for token in tokens])
        q.task_done()

def thread_stem_tokens(tokens):
    stemmed_tokens = []
    token_chunks = chunkify(tokens, 100000)
    q = Queue(maxsize=0)
    output_queue = Queue(maxsize=0)
    num_threads = 10

    for chunk in token_chunks:
        worker = Thread(target=do_stuff, args=(q, output_queue,))
        worker.setDaemon(True)
        worker.start()

    for chunk in token_chunks:
        q.put(chunk)

    q.join()

    while not output_queue.empty():
        stemmed_tokens.append(output_queue.get())

    return [item for sublist in stemmed_tokens for item in sublist]

def get_unigrams(sentence, get_base_words=False):
    global unigram_cache
    if not unigram_cache:
        log('caching unigrams...')
        log('tokenizing...')
        tokens = nltk.word_tokenize(sentence)

        if get_base_words:
            log('stemming...')
            # unigram_cache = stem_tokens(tokens)
            unigram_cache = multiprocess_stem_tokens(tokens)
            # unigram_cache = thread_stem_tokens(tokens)
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

def tag_cloud_to_file(tag_cloud, filename, sort_input=False):
    if sort_input:
        tag_cloud.sort()

    with open(filename, 'w', newline='') as file_path:
        writer = csv.writer(file_path)
        writer.writerows(tag_cloud)

def join_sentence_list(sentence_list):
    STOPWORDS = nltk.corpus.stopwords.words('english')
    stopwords_pattern = re.compile(r'\b(' + r'|'.join(STOPWORDS) + r')\b\s*')
    special_characters_pattern = re.compile(r'([^\s\w]|_)+')

    combined_string = (' ' + str(NEWLINE_CONSTANT) + ' ').join(sentence_list)
    combined_string = combined_string.lower()
    combined_string = stopwords_pattern.sub(' ', combined_string)
    combined_string = special_characters_pattern.sub('', combined_string)

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
    ngram_no_newlines_append = ngram_no_newlines.append
    newline = NEWLINE_CONSTANT
    for (ngram, count) in ngram_tagcloud:
        if newline not in ngram:
            ngram_no_newlines_append((ngram, count))

    output_path = DIRNAME + '/output/' + str(ntype) + '_ngram_tagcloud.csv'
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
