from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
from nltk import sent_tokenize, word_tokenize
import string
import math


def compute_tf(frequency, total_words_count):
    return frequency / total_words_count


def compute_idf(word, words_of_documents):
    return math.log10(len(words_of_documents) / sum(1 for words in words_of_documents if word in words))


def compute_tf_idf(word, word_frequency, words_of_documents):
    total_words_count = len(sum(words_of_documents, []))
    return compute_tf(word_frequency, total_words_count) * compute_idf(word, words_of_documents)


def input_compression_percentage():
    compression_percentage_correct = False
    compression_percent = 0
    while not compression_percentage_correct:
        compression_percent = float(input('Введите процент сжатия (1-100): '))
        if compression_percent > 100.0 or compression_percent < 0.5:
            print('Введите число от 1 до 100')
        else:
            compression_percentage_correct = True
    return compression_percent


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


def get_normal_form_words(words):
    return [MorphAnalyzer().parse(word)[0].normal_form for word in words]


def get_sentences(txt_path):
    input_file = open(txt_path, 'r', encoding='UTF-8')
    text = input_file.read().lower()
    return sent_tokenize(text)


def get_clean_tokens(sentence):
    tokens = word_tokenize(sentence)
    tokens = [t for t in tokens if t not in punctuation and t not in stop_words]
    return tokens


stop_words = stopwords.words('russian')
stop_words.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', 'к', 'на', '–', '-', '—', '«', '»'])
punctuation = string.punctuation
