import string
from collections import defaultdict, Counter
from nltk import word_tokenize, sent_tokenize, download
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
import main


if __name__ == '__main__':
    # download('punkt')
    # download('stopwords')
    input_file_path = 'input_1.txt'
    input_file = open(input_file_path, 'r', encoding='UTF-8')
    text = input_file.read().lower()
    sentences = sent_tokenize(text)
    stop_words = stopwords.words('russian')
    stop_words.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', 'к', 'на', '–', '-', '—', '«', '»'])
    frequency_dictionary_lem = defaultdict(int)

    for sentence in sentences:
        tokens = word_tokenize(sentence)
        tokens = [t for t in tokens if t not in string.punctuation and t not in stop_words]

        for token in tokens:
            norm_word = MorphAnalyzer().parse(token)[0].normal_form
            frequency_dictionary_lem[norm_word] += 1

    with open('freq_dict_lem.txt', 'w', encoding='UTF-8') as output_file:
        for token, appearances_count in main.sort_dictionary_by_value(frequency_dictionary_lem, reverse=True).items():
            output_file.write(f'{token}: {appearances_count}\n')
