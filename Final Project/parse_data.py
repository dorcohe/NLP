import csv
SemEval_prefix = "SemEval_"
import pandas as pd
import numpy as np
import pandas as pd
from sklearn import preprocessing

class Parse_data:

    def  __init__ (self, file_path, testing_emotions_path):
        self.file_path = file_path
        self.testing_emotions_path = testing_emotions_path
        self.sem_eval2007_id_to_emotions = self.get_sem_eval2007_id_to_emotions()
        self.train_data()

    def train_data(self):
        data_from_file = pd.read_csv(self.file_path)
        boolean_indexes = data_from_file["id"].str.startswith(SemEval_prefix)
        self.train_instances = data_from_file[~boolean_indexes]
        self.test_instances = data_from_file[boolean_indexes]

        extra_column = [np.nan] * len(self.test_instances)
        self.test_instances.insert(5, "Categorial", extra_column, True)

        for index in self.sem_eval2007_id_to_emotions:
            value = self.sem_eval2007_id_to_emotions[index]
            self.test_instances.loc[self.test_instances.id == SemEval_prefix + str(index), "Categorial"] = [value]

        self.test_instances = self.test_instances.dropna(axis=0, how='any')

        train_instances = list(range(len(self.train_instances)))
        self.train_instances.reindex(train_instances)

        self.normalize_data()


    def normalize_data(self):

        #Train
        mean = self.train_instances.mean()
        std = self.train_instances.std()

        current_v = self.train_instances["V"]
        normalized_v = (current_v - mean["V"]) / std["V"]
        self.train_instances["V"] = normalized_v

        current_a = self.train_instances["A"]
        normalized_a = (current_a - mean["A"]) / std["A"]
        self.train_instances["A"] = normalized_a

        current_d = self.train_instances["D"]
        normalized_d = (current_d - mean["D"]) / std["D"]
        self.train_instances["D"] = normalized_d

        #Test
        mean = self.test_instances.mean()
        std = self.test_instances.std()

        current_v = self.test_instances["V"]
        normalized_v = (current_v - mean["V"]) / std["V"]
        self.test_instances["V"] = normalized_v

        current_a = self.test_instances["A"]
        normalized_a = (current_a - mean["A"]) / std["A"]
        self.test_instances["A"] = normalized_a

        current_d = self.test_instances["D"]
        normalized_d = (current_d - mean["D"]) / std["D"]
        self.test_instances["D"] = normalized_d


    def get_train_data(self):

      return self.train_instances


    def get_id_to_sentence_dict(self):
        sentences = {}
        with open(self.file_path, encoding='utf8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    if line_count == 393:
                        line_count += 1
                    sentences.update({row[0]: row[1]})
                    line_count += 1
        return sentences

    def get_sem_eval2007_id_to_emotions(self):
        sentences = {}
        with open(self.testing_emotions_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')

            # row is in the form: 0: id  1: anger 2: disgust 3: fear 4: joy 5 :sadness 6: surprise
            for row in csv_reader:
                instance = {
                    "id": row[0],
                    "anger": row[1],
                    "disgust": row[2],
                    "fear": row[3],
                    "joy": row[4],
                    "sadness": row[5],
                    "surprise": row[6]
                }
                sentences.update({row[0]: instance})

        return sentences


    def get_test_data(self):
        return self.test_instances

# reader_path = "./corpus/reader.csv"
# writer_path = "./corpus/writer.csv"
# raw_path = "./corpus/raw.csv"
# sem2007_paths = "./affectivetext_test.emotions.gold"
#
# moss = parse_data(reader_path, writer_path, raw_path, sem2007_paths)
#
# a = moss.get_train_data()
# b = moss.get_test_data()
# a = 3