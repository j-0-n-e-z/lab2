from collections import defaultdict
from nltk import download
from main import get_sentences, get_clean_tokens, get_normal_form_words, sort_dictionary_by_value, compute_tf_idf
# TF термина а = (Количество раз, когда термин а встретился в тексте / количество всех слов в тексте)
# IDF термина а = log(Общее количество документов / Количество документов, в которых встречается термин а)
# Чем важнее слово для определения темы документа внутри коллекции, тем больше произведение
# keywords - первые 20 слов частотного словаря
if __name__ == '__main__':
    word_frequency = defaultdict(int)
    keyword_tf_idf = defaultdict(float)
    words_of_sentences = []
    keywords_count = 20

    for sentence in get_sentences('input.txt'):
        tokens = get_clean_tokens(sentence)
        norm_words = get_normal_form_words(tokens)
        words_of_sentences.append(norm_words)
        for norm_word in norm_words:
            word_frequency[norm_word] += 1

    word_frequency = sort_dictionary_by_value(word_frequency, reverse=True)

    for keyword, frequency in list(word_frequency.items())[:keywords_count]:
        keyword_tf_idf[keyword] = compute_tf_idf(keyword, frequency, words_of_sentences)

    with open('outputs/task_3_output/keyword_tf_idf.txt', 'w', encoding='UTF-8') as output_file:
        for keyword, tf_idf in sort_dictionary_by_value(keyword_tf_idf, reverse=True).items():
            output_file.write(f'{keyword}: {tf_idf}\n')




