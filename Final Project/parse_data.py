import csv
SemEval_prefix = "SemEval"

class Parse_data:

    def  __init__ (self, reader_path, writer_path, raw_information_path, sem2007_paths):
        self.reader_path = reader_path
        self.writer_path = writer_path
        self.raw_information_path = raw_information_path
        self.sem2007_paths = sem2007_paths
        self.id_to_sentence = self.get_id_to_sentence_dict()
        self.sem_eval2007_id_to_emotions = self.get_sem_eval2007_id_to_emotions()



    def normalize_data(self):

        #each instance has id, sentence, v, a, d
        v_instances = []
        a_instances = []
        d_instances = []
        with open(self.reader_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='Parse_data,')
            line_count = 0

            #row is in the form: 0: id, 1:v, 2:a, 3:d, 4:stdV, 5:stdA, 6:stdD, 7:N
            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    if not row[0].startswith(SemEval_prefix):
                            v_instances.append(float(row[1]))
                            a_instances.append(float(row[2]))
                            d_instances.append(float(row[3]))
                    line_count += 1
            print(f'Processed {line_count} lines.')
        moss = v_instances.mean()
        moss = v_instances.std()


    def get_train_data(self):

        #each instance has id, sentence, v, a, d
        instances = []
        with open(self.reader_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            #row is in the form: 0: id, 1:v, 2:a, 3:d, 4:stdV, 5:stdA, 6:stdD, 7:N
            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    if not row[0].startswith(SemEval_prefix):
                        instance = {
                            "id": row[0],
                            "text": self.id_to_sentence[row[0]],
                            "v": float(row[1]),
                            "a": float(row[2]),
                            "d": float(row[3])
                        }
                        instances.append(instance)
                    line_count += 1
            print(f'Processed {line_count} lines.')

        return instances

    def get_id_to_sentence_dict(self):
        sentences = {}
        with open(self.raw_information_path, encoding='utf8') as csv_file:
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
        with open(self.sem2007_paths) as csv_file:
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
        #each instance has id, sentence, v, a, d
        instances = []
        with open(self.reader_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            #row is in the form: 0: id, 1:v, 2:a, 3:d, 4:stdV, 5:stdA, 6:stdD, 7:N
            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    if row[0].startswith(SemEval_prefix):
                        # row[0] is in the format 'SemEval_{sentenceId}'
                        splited_id = row[0].split("_")
                        try:
                            instance = {
                                "id": row[0],
                                "text": self.id_to_sentence[row[0]],
                                "v": float(row[1]),
                                "a": float(row[2]),
                                "d": float(row[3]),
                                "categories": self.sem_eval2007_id_to_emotions[splited_id[1]]
                            }
                            instances.append(instance)
                        except:
                            #
                            a = 3
                    line_count += 1
            print(f'Processed {line_count} lines.')

        return instances

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