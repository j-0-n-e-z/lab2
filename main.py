import string
from collections import defaultdict, Counter
from nltk import word_tokenize, sent_tokenize, download
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from pymorphy2 import MorphAnalyzer
import math


def compute_tf(frequency, total_words_count):
    return frequency / total_words_count
# #На вход берем текст в виде списка (list) слов
#     #Считаем частотность всех терминов во входном массиве с помощью
#     #метода Counter библиотеки collections
#     tf_text = Counter(words)
#     for i in tf_text:
#         #для каждого слова в tf_text считаем TF путём деления
#         #встречаемости слова на общее количество слов в тексте
#         tf_text[i] = tf_text[i] / float(len(words))
#     #возвращаем объект типа Counter c TF всех слов текста
#     return tf_text


def compute_idf(word, words_of_sentences):
#на вход берется слово, для которого считаем IDF
#и корпус документов в виде списка списков слов
        #количество документов, где встречается искомый термин
        #считается как генератор списков
    return math.log10(len(words_of_sentences) / sum([1.0 for words in words_of_sentences if word in words]))


def compute_tf_idf(word, word_frequency, words_of_sentences):
    total_words_count = len(sum(words_of_sentences, []))
    return compute_tf(word_frequency, total_words_count) * compute_idf(word, words_of_sentences)


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


# download('stopwords')
compression_percentage = input_compression_percentage()
input_file_path = 'input_1.txt'
input_file = open(input_file_path, 'r', encoding='UTF-8')
text = input_file.read().lower()
sentences = sent_tokenize(text)

snowball = SnowballStemmer('russian')
stop_words = stopwords.words('russian')
stop_words.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', 'к', 'на', '–', '-', '—', '«', '»'])

frequency_dictionary_lem = defaultdict(int)
frequency_dictionary_stem = defaultdict(int)
sentences_dictionary = defaultdict(int)
norm_words_of_sentences = []

print('Обработка текста')
for sentence in sentences:
    tokens = word_tokenize(sentence)
    tokens = [t for t in tokens if t not in string.punctuation and t not in stop_words]
    norm_words = []

    for token in tokens:
        norm_word = MorphAnalyzer().parse(token)[0].normal_form
        frequency_dictionary_lem[norm_word] += 1
        frequency_dictionary_stem[snowball.stem(token)] += 1
        norm_words.append(norm_word)

    norm_words_of_sentences.append(norm_words)

sorted_frequency_dictionary_lem = sort_dictionary_by_value(frequency_dictionary_lem, reverse=True)
with open('freq_dict_lem.txt', 'w', encoding='UTF-8') as output_file:
    for token, appearances_count in sorted_frequency_dictionary_lem.items():
        output_file.write(f'{token}: {appearances_count}\n')

with open('freq_dict_stem.txt', 'w', encoding='UTF-8') as output_file:
    for token, appearances_count in sort_dictionary_by_value(frequency_dictionary_stem, reverse=True).items():
        output_file.write(f'{token}: {appearances_count}\n')

# task 4
# get sentences freq
word_tf_idf = defaultdict(float)
for norm_words in norm_words_of_sentences:
    for norm_word in norm_words:
        word_tf_idf[norm_word] = compute_tf_idf(norm_word, frequency_dictionary_lem[norm_word], norm_words_of_sentences)

with open(f'word_tf_idf.txt', 'w', encoding='UTF-8') as output_file:
    for word, tf_idf in sort_dictionary_by_value(word_tf_idf, reverse=True).items():
        output_file.write(f'{word}: {tf_idf}\n')

for sentence_idx, norm_words in enumerate(norm_words_of_sentences):
    for norm_word in norm_words:
        sentences_dictionary[sentences[sentence_idx]] += word_tf_idf[norm_word]
# save sentences freq
sorted_sentence_dictionary = sort_dictionary_by_value(sentences_dictionary, reverse=True)
with open(f'sentences_frequency.txt', 'w', encoding='UTF-8') as output_file:
    for sentence, frequency in sorted_sentence_dictionary.items():
        output_file.write(f'{sentence}: {frequency}\n')
# get sentences with compression
sorted_sentences = list(sorted_sentence_dictionary.keys())
taken_sentences_count = round(len(sorted_sentences) * compression_percentage / 100)
sorted_sentences_compressed = sorted_sentences[:taken_sentences_count]
# get taken sentences in the order of appearance
compressed_sentences_with_index = {}
for sentence in sorted_sentences_compressed:
    compressed_sentences_with_index[sentence] = sentences.index(sentence)
sorted_report_sentences = sort_dictionary_by_value(compressed_sentences_with_index, reverse=False)
report_sentences = [sentence.capitalize() for sentence in sorted_report_sentences.keys()]
# save report
with open(f'report_{compression_percentage}.txt', 'w', encoding='UTF-8') as output_file:
    for sentence in report_sentences:
        output_file.write(sentence + '\n')

# keywords = list(frequency_dictionary_lem.items())[:20]
# print(compute_idf(keywords[0], norm_words_of_sentences))
#
# keywords = list(frequency_dictionary_lem.keys())[:20]
# idfs = []
# for keyword in keywords:
#     idfs.append(compute_idf(keyword, norm_words_of_sentences))
#
# total_words_count = 0
# for words in norm_words_of_sentences:
#     total_words_count += len(words)
# keywords_tuples = list(frequency_dictionary_lem.values())[:20]
# compute_tf(keywords_tuples, total_words_count)

keywords = list(sorted_frequency_dictionary_lem.items())[:20]
keyword_tf_idf = defaultdict(float)
for keyword_tuple in keywords:
    tf_idf = compute_tf_idf(keyword_tuple[0], keyword_tuple[1], norm_words_of_sentences)
    keyword_tf_idf[keyword_tuple[0]] = tf_idf

with open(f'keyword_tf_idf.txt', 'w', encoding='UTF-8') as output_file:
    for keyword, tf_idf in keyword_tf_idf.items():
        output_file.write(f'{keyword}: {tf_idf}\n')
