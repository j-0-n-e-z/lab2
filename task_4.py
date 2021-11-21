from collections import defaultdict
from nltk import download
from main import *


if __name__ == '__main__':
    word_frequency = defaultdict(int)
    word_tf_idf = defaultdict(float)
    sentence_frequency = defaultdict(float)
    sentence_words = defaultdict(list)
    words_of_sentences = []

    compression_percentage = get_compression_percentage()
    sentences = get_sentences('input.txt')

    for sentence in sentences:
        tokens = get_clean_tokens(sentence)
        norm_words = get_normal_form_words(tokens)
        sentence_words[sentence] = norm_words
        words_of_sentences.append(norm_words)
        for norm_word in norm_words:
            word_frequency[norm_word] += 1

    word_frequency = sort_dictionary_by_value(word_frequency, reverse=True)

    all_words_of_text = sum(words_of_sentences, [])
    for word in all_words_of_text:
        word_tf_idf[word] = compute_tf_idf(word, word_frequency[word], words_of_sentences)

    word_tf_idf = sort_dictionary_by_value(word_tf_idf, reverse=True)

    with open('outputs/task_4_output/word_tf_idf.txt', 'w', encoding='UTF-8') as output:
        for word, tf_idf in word_tf_idf.items():
            output.write(f'{word}: {tf_idf}\n')

    for sentence, norm_words in sentence_words.items():
        for norm_word in norm_words:
            sentence_frequency[sentence] += word_tf_idf[norm_word]

    sentence_frequency = sort_dictionary_by_value(sentence_frequency, reverse=True)

    with open('outputs/task_4_output/sentence_frequency.txt', 'w', encoding='UTF-8') as output_file:
        for sentence, frequency in sentence_frequency.items():
            output_file.write(f'{sentence}: {frequency}\n')

    sentences_sorted_by_frequency = list(sentence_frequency.keys())
    taken_sentences_count = round(len(sentences_sorted_by_frequency) * compression_percentage / 100)
    compressed_sentences = sentences_sorted_by_frequency[:taken_sentences_count]

    sentence_index_dictionary = {}
    for sentence in compressed_sentences:
        sentence_index_dictionary[sentence] = sentences.index(sentence)

    sentence_index_dictionary = sort_dictionary_by_value(sentence_index_dictionary, reverse=False)

    report_sentences = [sentence.capitalize() for sentence in sentence_index_dictionary.keys()]

    with open(f'outputs/task_4_output/report_{compression_percentage}.txt', 'w', encoding='UTF-8') as output_file:
        for sentence in report_sentences:
            output_file.write(sentence + '\n')
