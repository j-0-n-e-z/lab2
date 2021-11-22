from nltk import download
from main import *


if __name__ == '__main__':
    sentence_frequency = defaultdict(float)
    sentence_words = defaultdict(list)

    compression_percentage = get_compression_percentage()
    sentences = get_sentences('input.txt')
    word_frequency_counter = get_word_counter_with_lemmatization(sentences)

    for sentence in sentences:
        tokens = get_clean_tokens(sentence)
        norm_words = get_normal_form_words(tokens)
        sentence_words[sentence] = norm_words

    word_tf_idf = compute_words_tf_idf(list(sentence_words.values()), word_frequency_counter)

    for sentence, norm_words in sentence_words.items():
        for word in norm_words:
            sentence_frequency[sentence] += word_tf_idf[word]

    sentence_frequency = sort_dictionary_by_value(sentence_frequency, reverse=True)

    sentences_sorted_by_frequency = list(sentence_frequency.keys())
    compressed_sentences_count = round(len(sentences_sorted_by_frequency) * compression_percentage / 100)
    compressed_sentences = sentences_sorted_by_frequency[:compressed_sentences_count]

    sentence_index_dictionary = {}
    for sentence in compressed_sentences:
        sentence_index_dictionary[sentence] = sentences.index(sentence)

    sentence_index_dictionary = sort_dictionary_by_value(sentence_index_dictionary, reverse=False)
    report_sentences = [sentence.capitalize() for sentence in sentence_index_dictionary.keys()]

    with open('outputs/task_4_output/word_tf_idf.txt', 'w', encoding='UTF-8') as output_file:
        for word, tf_idf in word_tf_idf.items():
            output_file.write(f'{word}: {tf_idf}\n')

    with open('outputs/task_4_output/sentence_frequency.txt', 'w', encoding='UTF-8') as output_file:
        for sentence, frequency in sentence_frequency.items():
            output_file.write(f'{sentence}: {frequency}\n')

    with open(f'outputs/task_4_output/report_{compression_percentage}.txt', 'w', encoding='UTF-8') as output_file:
        for sentence in report_sentences:
            output_file.write(sentence + '\n')
