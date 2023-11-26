import re
import pandas as pd
import sqlite3
from pymorphy3 import MorphAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

class DatabaseClass:
    def __init__(self):
        # Ініціалізація  об'єкта бази даних
        self._cursor = sqlite3.connect('database.db').cursor()

    def select_word_avg_freq(self):
        #виведення для кількості тегів частин мови у pos_frequencies
        query_total_pos_tags = '''
            SELECT COUNT(pos_tag) AS total_pos_tags
            FROM pos_frequencies;
        '''
        result_total_pos_tags = self._cursor.execute(query_total_pos_tags)
        print("\nTotal POS Tags:")
        print(result_total_pos_tags.fetchone()[0])

        #виведення для кількості лем у lemma_frequencies
        query_total_lemmas = '''
            SELECT COUNT(lemma) AS total_lemmas
            FROM lemma_frequencies;
        '''
        result_total_lemmas = self._cursor.execute(query_total_lemmas)
        print("\nTotal Lemmas:")
        print(result_total_lemmas.fetchone()[0])

        #виведення для кількості унікальних слів у word_frequencies
        query_unique_words = '''
            SELECT COUNT(DISTINCT word) AS unique_words
            FROM word_frequencies;
        '''
        result_unique_words = self._cursor.execute(query_unique_words)
        print("\nUnique Words:")
        print(result_unique_words.fetchone()[0])

        #виведення для топ-10 слів за кількістю вживань
        query_top_words = '''
            SELECT
                word,
                SUM(freq_1) AS total_usage
            FROM
                word_frequencies
            GROUP BY
                word
            ORDER BY
                total_usage DESC
            LIMIT 10;
        '''
        result_top_words = self._cursor.execute(query_top_words)
        print("\nTop 10 Words:")
        for row in result_top_words:
            print(row)

        #виведення для топ-10 лем за кількістю вживань
        query_top_lemmas = '''
            SELECT
                lemma,
                SUM(freq_1) AS total_usage
            FROM
                lemma_frequencies
            GROUP BY
                lemma
            ORDER BY
                total_usage DESC
            LIMIT 10;
        '''
        result_top_lemmas = self._cursor.execute(query_top_lemmas)
        print("\nTop 10 Lemmas:")
        for row in result_top_lemmas:
            print(row)


filename = input("Enter the file name (with its extension) you want to process: ")

with open(filename, encoding="utf-8") as data_1:
    text1 = data_1.read().lower()

    # Використовуємо регулярний вираз для токенізації слів
    tokens = re.findall(r'\b\w+(?:[-\'’]\w+)*\b', text1)

    # Обмежуємо кількість токенів до 20000
    tokens = tokens[:20000]

    # Розбиваємо на 20 груп по 1000 слів
    chunk_size = 1000
    num_chunks = len(tokens) // chunk_size
    token_groups = [tokens[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks)]

    # Створюємо частотний словник для кожної вибірки
    frequency_dicts = []
    for i, group in enumerate(token_groups):
        frequency_dict = {}
        for word in group:
            frequency_dict[word] = frequency_dict.get(word, 0) + 1
        frequency_dicts.append(frequency_dict)

    # Створюємо DataFrame для токенів
    df = pd.DataFrame(frequency_dicts).fillna(0).astype(int).T
    df.columns = [f'freq_{i + 1}' for i in range(num_chunks)]

    # Замінюємо індекси на 'word'
    df.index.name = 'word'
    df.reset_index(inplace=True)

    # З'єднуємося з базою даних SQLite
    conn = sqlite3.connect('database.db')

    # Записуємо DataFrame в базу даних
    df.to_sql('word_frequencies', conn, if_exists='replace', index=False)

    # Ініціалізуємо морфологічний аналізатор для української мови
    morph = MorphAnalyzer(lang='uk')

    # Функція для отримання леми слова
    def get_lemma(word):
        parsed_word = morph.parse(word)[0]
        return parsed_word.normal_form

    # Функція для отримання частини мови слова
    def get_pos(word):
        parsed_word = morph.parse(word)[0]
        return parsed_word.tag.POS

    # Застосовуємо функції до всіх слів у тексті
    lemmas = [get_lemma(word) for word in tokens]
    pos_tags = [get_pos(word) for word in tokens]

    # Розбиваємо на 20 груп по 1000 лем та тегів частин мови
    lemma_groups = [lemmas[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks)]
    pos_groups = [pos_tags[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks)]

    # Створюємо частотні словники для кожної вибірки лем та тегів частин мови
    lemma_frequency_dicts = []
    pos_frequency_dicts = []

    for i, (lemma_group, pos_group) in enumerate(zip(lemma_groups, pos_groups)):
        lemma_frequency_dict = {}
        pos_frequency_dict = {}

        for lemma, pos_tag in zip(lemma_group, pos_group):
            lemma_frequency_dict[lemma] = lemma_frequency_dict.get(lemma, 0) + 1
            pos_frequency_dict[pos_tag] = pos_frequency_dict.get(pos_tag, 0) + 1

        lemma_frequency_dicts.append(lemma_frequency_dict)
        pos_frequency_dicts.append(pos_frequency_dict)

    # Створюємо DataFrame для частот лем та тегів частин мови
    lemma_df = pd.DataFrame(lemma_frequency_dicts).fillna(0).astype(int).T
    pos_df = pd.DataFrame(pos_frequency_dicts).fillna(0).astype(int).T

    # Додаємо префікси до назв колонок для легшого визначення групи
    lemma_df.columns = [f'freq_{i + 1}' for i in range(num_chunks)]
    pos_df.columns = [f'freq_{i + 1}' for i in range(num_chunks)]

    # Замінюємо індекси на 'lemma' та 'pos_tag'
    lemma_df.index.name = 'lemma'
    pos_df.index.name = 'pos_tag'

    lemma_df.reset_index(inplace=True)
    pos_df.reset_index(inplace=True)

    # Записуємо DataFrame в базу даних
    lemma_df.to_sql('lemma_frequencies', conn, if_exists='replace', index=False)
    pos_df.to_sql('pos_frequencies', conn, if_exists='replace', index=False)

    # Додаємо колонку 'total_pos_frequencies' до таблиці 'pos_frequencies'
    query = """
        ALTER TABLE pos_frequencies
        ADD COLUMN total_pos_frequencies INTEGER
    """

    conn.execute(query)

    # Підрахунок абсолютної частоти для кожної частини мови
    for i in range(1, num_chunks + 1):
        query = f"""
            UPDATE pos_frequencies
            SET total_pos_frequencies = COALESCE(freq_{i}, 0) + COALESCE(total_pos_frequencies, 0)
        """
        conn.execute(query)

    # Розрахунок TF-IDF з обмеженням кількості фіч
    vectorizer = TfidfVectorizer(max_features=1000)  # Adjust the number as needed
    tfidf_matrix = vectorizer.fit_transform([' '.join(lemmas)])

    # Створюємо DataFrame для TF-IDF
    tfidf_df = pd.DataFrame(list(zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0])),
                            columns=['word', 'tfidf'])
    # Записуємо DataFrame в базу даних
    tfidf_df.to_sql('tfidf', conn, if_exists='replace', index=False)

    # Викликаємо метод select_word_avg_freq
    database_instance = DatabaseClass()
    database_instance.select_word_avg_freq()

    # Закриваємо з'єднання
    conn.close()
