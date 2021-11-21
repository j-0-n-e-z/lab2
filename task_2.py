from collections import defaultdict
from nltk import download
from nltk.stem import SnowballStemmer
from main import get_sentences, get_clean_tokens, sort_dictionary_by_value


if __name__ == '__main__':
    word_frequency = defaultdict(int)
    snowball = SnowballStemmer('russian')

    for sentence in get_sentences('input.txt'):
        for token in get_clean_tokens(sentence):
            word_frequency[snowball.stem(token)] += 1

    with open('outputs/task_2_output/frequency_dictionary_stemming.txt', 'w', encoding='UTF-8') as output_file:
        for word, frequency in sort_dictionary_by_value(word_frequency, reverse=True).items():
            output_file.write(f'{word}: {frequency}\n')
