from nltk import download
from main import *


if __name__ == '__main__':
    keyword_tf_idf = defaultdict(float)
    words_of_sentences = []
    keywords_count = 20

    sentences = get_sentences('input.txt')
    word_frequency_counter = get_word_counter_with_lemmatization(sentences)

    for sentence in sentences:
        tokens = get_clean_tokens(sentence)
        norm_words = get_normal_form_words(tokens)
        words_of_sentences.append(norm_words)

    for keyword, frequency in word_frequency_counter[:keywords_count]:
        keyword_tf_idf[keyword] = compute_tf_idf(keyword, frequency, words_of_sentences)

    with open('outputs/task_3_output/keyword_tf_idf.txt', 'w', encoding='UTF-8') as output_file:
        for keyword, tf_idf in sort_dictionary_by_value(keyword_tf_idf, reverse=True).items():
            output_file.write(f'{keyword}: {tf_idf}\n')
