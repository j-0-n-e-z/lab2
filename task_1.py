from collections import defaultdict
from nltk import download
from main import get_sentences, get_clean_tokens, get_normal_form_words, sort_dictionary_by_value


if __name__ == '__main__':
    word_frequency = defaultdict(int)

    for sentence in get_sentences('input.txt'):
        tokens = get_clean_tokens(sentence)
        for norm_word in get_normal_form_words(tokens):
            word_frequency[norm_word] += 1

    with open('outputs/task_1_output/frequency_dictionary_lemmatization.txt', 'w', encoding='UTF-8') as output_file:
        for word, frequency in sort_dictionary_by_value(word_frequency, reverse=True).items():
            output_file.write(f'{word}: {frequency}\n')
