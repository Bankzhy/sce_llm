import math
import re

import numpy as np

from reflect.stop_word_remover import StopWordRemover


class TFIDF:
    def __init__(self):
        self.method_list = []
        self.processed_method_list = []
        self.unique_statements = []
        self.tfIdf_vectors = []
        self.euclidean_norm = []

    def calc_with_statements(self, method_list):
        self.method_list = method_list
        self.processed_method_list = self.process_with_statement()
        self.calculate_vectors()

    def calculate_vectors(self):
        unique_statements = []
        for statement_list in self.processed_method_list:
            unique_statement = list(set(statement_list))
            unique_statements.extend(unique_statement)
        num_unique_statements = len(unique_statements)
        self.euclidean_norm = [0*i for i in range(len(self.processed_method_list))]
        m_num = 0
        self.tfIdf_vectors = np.zeros((len(self.processed_method_list), num_unique_statements))
        for statement_list in self.processed_method_list:
            tf = 0
            idf = 0
            st_num = 0
            tf_idf_vec = [0*i for i in range(num_unique_statements)]
            for statement in unique_statements:
                tf = self.get_tf(statement, statement_list)
                idf = self.get_idf(statement)
                tf_idf_vec[st_num] = tf*idf
                st_num += 1

            self.tfIdf_vectors[m_num] = tf_idf_vec
            en = self.calculate_euclidean_norm(tf_idf_vec)
            self.euclidean_norm[m_num] = en

            m_num += 1
        # print(tf_idf_vec)

    def process_with_statement(self):
        processed_method_list = []
        for method in self.method_list:
            process_method = self.remove_stop_word(method.word_list)
            process_method = self.remove_special_char(process_method)
            process_method = self.split_token(process_method)
            process_method = self.to_lower(process_method)
            # ps = PorterStemmer()
            # process_method = [ps.stem(o) for o in process_method]
            processed_method_list.append(process_method)
            # print(process_method)
        return processed_method_list

    def to_lower(self, word_list):
        result = []
        for word in word_list:
            result.append(word.lower())
        return result

    def split_token(self, word_list):
        result = []
        for word in word_list:
            ws = self.camel_case_split(word)
            result.extend(ws)
        result = self.under_case_split(result)
        return result

    def under_case_split(self, word_list):
        result = []
        for w in word_list:
            ws = w.split("_")
            result.extend(ws)
        return result

    def camel_case_split(self, str):
        if len(str) == 0:
            return ""
        words = [[str[0]]]

        for c in str[1:]:
            if words[-1][-1].islower() and c.isupper():
                words.append(list(c))
            else:
                words[-1].append(c)

        return [''.join(word) for word in words]

    def remove_stop_word(self, word_list):
        result_list = StopWordRemover().remove_stop_word(word_list=word_list)
        return result_list

    def remove_special_char(self, word_list):
        result_list = []
        pattern = "[^A-Za-z]+"
        for word in word_list:
            match_obj = re.match(pattern, word, re.M | re.I)
            if match_obj is None:
                result_list.append(word)
        return result_list

    def get_tf(self, statement, statement_list):
        count = 0
        for st in statement_list:
            if st == statement:
                count += 1
        return count

    def get_idf(self, statement):
        count = 0
        for statement_list in self.processed_method_list:
            if statement in statement_list:
                count += 1
        result = math.log(len(self.processed_method_list) / count)
        return result

    def calculate_euclidean_norm(self, vec):
        norm = 0
        for v in vec:
            norm += v * v
        return math.sqrt(norm)