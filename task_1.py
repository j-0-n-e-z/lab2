from nltk import download
from main import get_sentences, get_word_counter_with_lemmatization


if __name__ == '__main__':
    sentences = get_sentences('input.txt')
    word_frequency_counter = get_word_counter_with_lemmatization(sentences)

    with open('outputs/task_1_output/frequency_dictionary_lemmatization.txt', 'w', encoding='UTF-8') as output_file:
        for word, frequency in word_frequency_counter:
            output_file.write(f'{word}: {frequency}\n')
