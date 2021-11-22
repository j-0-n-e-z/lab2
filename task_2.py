from nltk import download
from main import get_sentences, get_word_counter_with_stemming


if __name__ == '__main__':
    sentences = get_sentences('input.txt')
    word_frequency_counter = get_word_counter_with_stemming(sentences)

    with open('outputs/task_2_output/frequency_dictionary_stemming.txt', 'w', encoding='UTF-8') as output_file:
        for word, frequency in word_frequency_counter:
            output_file.write(f'{word}: {frequency}\n')
