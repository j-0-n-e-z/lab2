from collections import Counter, defaultdict
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
from nltk import sent_tokenize, word_tokenize
from nltk.stem import SnowballStemmer
import string
import math


def compute_tf(frequency, total_words_count):
    return frequency / total_words_count


def compute_idf(word, words_of_documents):
    return math.log10(len(words_of_documents) / sum(1 for words in words_of_documents if word in words))


def compute_tf_idf(word, word_frequency, words_of_documents):
    total_words_count = len(sum(words_of_documents, []))
    return compute_tf(word_frequency, total_words_count) * compute_idf(word, words_of_documents)


def compute_words_tf_idf(words_of_sentences, word_frequency_counter):
    word_tf_idf = defaultdict(float)
    for word in sum(words_of_sentences, []):
        word_frequency = get_word_frequency_from_counter(word, word_frequency_counter)
        word_tf_idf[word] = compute_tf_idf(word, word_frequency, words_of_sentences)

    word_tf_idf = sort_dictionary_by_value(word_tf_idf, reverse=True)
    return word_tf_idf


def sort_dictionary_by_value(dictionary, reverse=False):
    return dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=reverse))


def get_compression_percentage():
    compression_percentage_correct = False
    compression_percent = 0
    while not compression_percentage_correct:
        compression_percent = float(input('Введите процент сжатия (1-100): '))
        if compression_percent > 100 or compression_percent < 1:
            print('Введите число от 1 до 100')
        else:
            compression_percentage_correct = True
    return compression_percent


def get_normal_form_words(words: list):
    return [MorphAnalyzer().parse(word)[0].normal_form for word in words]


def get_sentences(txt_path: str):
    input_file = open(txt_path, 'r', encoding='UTF-8')
    text = input_file.read().lower()
    return sent_tokenize(text)


def get_clean_tokens(sentence: str):
    tokens = word_tokenize(sentence)
    tokens = [t for t in tokens if t not in punctuation and t not in stop_words]
    return tokens


def get_word_counter_with_lemmatization(sentences: list):
    tokens = sum([get_clean_tokens(sentence) for sentence in sentences], [])
    norm_words = get_normal_form_words(tokens)
    return Counter(norm_words).most_common()


def get_word_counter_with_stemming(sentences: list):
    snowball = SnowballStemmer('russian')
    tokens = sum([get_clean_tokens(sentence) for sentence in sentences], [])
    stems = [snowball.stem(token) for token in tokens]
    return Counter(stems).most_common()


def get_word_frequency_from_counter(word: str, counter: Counter):
    return list(filter(lambda t: t[0] == word, counter))[0][1]


stop_words = stopwords.words('russian')
stop_words.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', 'к', 'на', '–', '-', '—', '«', '»'])
punctuation = string.punctuation
