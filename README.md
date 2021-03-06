# Лабораторная работа #2

### Дисциплина: 
Технологии анализа данных в сети Интернет, 4 курс, НИУ ВШЭ г. Пермь

## Задание:
1. Написать программу (Python, С# или любой другой язык программирования) для построения частотного словаря текста на русском языке без учета стоп-слов. Входной текст хранится в текстовом файле, построенный словарь также должен быть сохранен в текстовый файл в формате (словоформа, количество_вхождений_словоформы) с сортировкой по количеству вхождений. Для деления текста на токены и удаления стоп-слов использовать библиотеку nltk, а для нормализации использовать библиотеку Pymorphy2. (2 балл)
2. Реализовать вторую версию программы построения чпстотного словаря, в которой для нормализации используется стеммер (найти необходимую библиотеку самостоятельно). Сравнить построенные частотные словари. (1 балла)
3. Собрать корпус документов по выбранной теме и по основы программ, разработанных в заданиях 1 и 2, подсчитать метрику TF-IDF для ключевых слов документа. (2 балла)
4. Реализовать алгоритм автоматического реферирования (квазиреферирование) на основе статистического подхода, алгоритм приведен ниже. Входные данные: исходный текст и коэффициент сжатия. Единицей реферирования (фрагментом) должно являться предложение. Выходные данные: список ключевых слов с весами, список предложений с весами, текст реферата. (5 баллов)

**Алгоритм**:
1. Разбить текст на предложения.
2. Разбить текст на слова произвести их нормализацию.
3. Удалить стоп-слова.
4. Подсчитать веса слов (частота слова в тексте или TF-IDF, если выполнялось задание 2).
5. Определить веса предложений, рассчитанный как сумма весов, входящих в предложение слов.
6. Отсортировать предложения по убыванию веса.
7. В отсортированном списке оставить те предложений, которые входят в задаваемый процент сжатия. Например, если процент сжатия 10%, а в исходном тексте 50 предложений, то в реферате будет 5 предложений.
8. Сформировать текст реферата из отобранных на предыдущем шаге предложений в порядке, котором они встретились в исходном тексте.
