import string
from collections import defaultdict, Counter
from nltk import word_tokenize, sent_tokenize, download
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
import math
import main


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


compression_percentage = input_compression_percentage()
full_text = ''
paths_of_input_files = ['input_1.txt', 'input_2.txt', 'input_3.txt', 'input_4.txt', 'input_5.txt']
for path in paths_of_input_files:
    input_file = open(path, 'r', encoding='UTF-8')
    full_text += input_file.read().lower()

sentences = sent_tokenize(full_text)
stop_words = stopwords.words('russian')
stop_words.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', 'к', 'на', '–', '-', '—', '«', '»'])

frequency_dictionary = defaultdict(int)
sentences_dictionary = defaultdict(int)
norm_words_of_sentences = []

print('Подсчет TF-IDF слов')
for sentence in sentences:
    words = word_tokenize(sentence)
    tokens = [w for w in words if w not in string.punctuation and w not in stop_words]
    norm_words = []

    for token in tokens:
        norm_word = MorphAnalyzer().parse(token)[0].normal_form
        frequency_dictionary[norm_word] += 1
        norm_words.append(norm_word)

    norm_words_of_sentences.append(norm_words)

frequency_dictionary = main.sort_dictionary_by_value(frequency_dictionary, reverse=True)

word_tf_idf = defaultdict(float)
for word in sum(norm_words_of_sentences, []):
    word_tf_idf[word] = main.compute_tf_idf(word, frequency_dictionary[word], norm_words_of_sentences)

word_tf_idf = main.sort_dictionary_by_value(word_tf_idf, reverse=True)
with open('word_tf_idf.txt', 'w', encoding='UTF-8') as output:
    for word, tf_idf in word_tf_idf.items():
        output.write(f'{word}: {tf_idf}\n')

# NOTE: Чем важнее слово для определения темы документа внутри коллекции, тем больше произведение

print('Подсчет весов предложений')
for sentence_idx, norm_words in enumerate(norm_words_of_sentences):
    for norm_word in norm_words:
        sentences_dictionary[sentences[sentence_idx]] += word_tf_idf[norm_word]

sorted_sentence_dictionary = sort_dictionary_by_value(sentences_dictionary, reverse=True)
with open(f'sentence_frequency.txt', 'w', encoding='UTF-8') as output_file:
    for sentence, frequency in sorted_sentence_dictionary.items():
        output_file.write(f'{sentence}: {frequency}\n')

sorted_sentences = list(sorted_sentence_dictionary.keys())
taken_sentences_count = round(len(sorted_sentences) * compression_percentage / 100)
sorted_sentences_compressed = sorted_sentences[:taken_sentences_count]

report_sentences_with_index = {}
for sentence in sorted_sentences_compressed:
    report_sentences_with_index[sentence] = sentences.index(sentence)

sorted_report_sentences = sort_dictionary_by_value(report_sentences_with_index, reverse=False)
report_sentences = [sentence.capitalize() for sentence in sorted_report_sentences.keys()]
with open(f'report_{compression_percentage}.txt', 'w', encoding='UTF-8') as output_file:
    for sentence in report_sentences:
        output_file.write(sentence + '\n')

print('Done')
