import string
from collections import defaultdict
from nltk import word_tokenize, sent_tokenize, download
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
from nltk.stem import SnowballStemmer
import math
import main


if __name__ == '__main__':
    full_text = ''
    paths_of_input_files = ['input_1.txt', 'input_2.txt', 'input_3.txt', 'input_4.txt', 'input_5.txt']
    texts_of_input_files = ['', '', '', '', '']
    for i, path in enumerate(paths_of_input_files):
        input_file = open(path, 'r', encoding='UTF-8')
        input_file_text = input_file.read().lower()
        texts_of_input_files[i] = input_file_text
        full_text += input_file_text

    sentences = sent_tokenize(full_text)
    stop_words = stopwords.words('russian')
    stop_words.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', 'к', 'на', '–', '-', '—', '«', '»'])
    words_of_input_files = []
    frequency_dictionary = defaultdict(int)

    for text in texts_of_input_files:
        sentences = sent_tokenize(text)
        words_of_text = []
        for sentence in sentences:
            tokens = [t for t in word_tokenize(sentence) if t not in string.punctuation and t not in stop_words]
            norm_words = [MorphAnalyzer().parse(t)[0].normal_form for t in tokens]
            words_of_text.extend(norm_words)
            for word in norm_words:
                frequency_dictionary[word] += 1
        words_of_input_files.append(words_of_text)

    words_count = len(sum(words_of_input_files, []))

    # TF термина а = (Количество раз, когда термин а встретился в тексте / количество всех слов в тексте)
    # IDF термина а = log(Общее количество документов / Количество документов, в которых встречается термин а)


    # for sentence in sentences:
    #     tokens = word_tokenize(sentence)
    #     tokens = [t for t in tokens if t not in string.punctuation and t not in stop_words]
    #
    #     for token in tokens:
    #         norm_word = MorphAnalyzer().parse(token)[0].normal_form
    #         frequency_dictionary[norm_word] += 1

    # frequency_dictionary = main.sort_dictionary_by_value(frequency_dictionary, reverse=True)
    # for word, freq in frequency_dictionary.items():
    #     print(f'{word}: {freq}')
    frequency_dictionary = main.sort_dictionary_by_value(frequency_dictionary, reverse=True)

    with open('freq.txt', 'w', encoding='UTF-8') as file:
        for word, freq in frequency_dictionary.items():
            file.write(f'{word}: {freq}')

    keywords = list(frequency_dictionary.items())[:20]
    keyword_tf_idf = defaultdict(float)
    for keyword_tuple in keywords:
        keyword = keyword_tuple[0]
        keyword_frequency = keyword_tuple[1]
        keyword_tf_idf[keyword] = main.compute_tf_idf(keyword, keyword_frequency, words_of_input_files)

    with open('keyword_tf_idf.txt', 'w', encoding='UTF-8') as output_file:
        for keyword, tf_idf in main.sort_dictionary_by_value(keyword_tf_idf, reverse=True).items():
            output_file.write(f'{keyword}: {tf_idf}\n')




