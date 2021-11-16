import string
from collections import defaultdict, Counter
from nltk import word_tokenize, sent_tokenize, download
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from pymorphy2 import MorphAnalyzer
import math


def compute_tf(words):
#На вход берем текст в виде списка (list) слов
    #Считаем частотность всех терминов во входном массиве с помощью
    #метода Counter библиотеки collections
    tf_text = Counter(words)
    for i in tf_text:
        #для каждого слова в tf_text считаем TF путём деления
        #встречаемости слова на общее количество слов в тексте
        tf_text[i] = tf_text[i] / float(len(words))
    #возвращаем объект типа Counter c TF всех слов текста
    return tf_text


def compute_idf(word, words):
#на вход берется слово, для которого считаем IDF
#и корпус документов в виде списка списков слов
        #количество документов, где встречается искомый термин
        #считается как генератор списков
    return math.log10(len(words) / sum([1.0 for i in words if word in i]))


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

with open('freq_dict_lem.txt', 'w', encoding='UTF-8') as output_file:
    for token, appearances_count in sort_dictionary_by_value(frequency_dictionary_lem, reverse=True).items():
        output_file.write(f'{token}: {appearances_count}\n')

with open('freq_dict_stem.txt', 'w', encoding='UTF-8') as output_file:
    for token, appearances_count in sort_dictionary_by_value(frequency_dictionary_stem, reverse=True).items():
        output_file.write(f'{token}: {appearances_count}\n')

# task 4
# get sentences freq
for sentence_idx, norm_words in enumerate(norm_words_of_sentences):
    for norm_word in norm_words:
        sentences_dictionary[sentences[sentence_idx]] += frequency_dictionary_lem[norm_word]
sorted_sentence_dictionary = sort_dictionary_by_value(sentences_dictionary, reverse=True)
# save sentences freq
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