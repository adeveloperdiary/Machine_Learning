import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans

data_folder = "HMP_Dataset/data/"
dirs = os.listdir(data_folder)

y_labels = {}
y_index = {}

random_state = 2
total_classes = len(dirs)


class VCClassifier:

    def __init__(self, k, segment_size, overlap, random_seg=0, enable_overlap=False):
        self.SEG_SAMPLE_SIZE = segment_size
        self.SEG_OVERLAP = overlap
        self.RANDOM_SEG = random_seg
        self.X_Dict = np.empty((0, self.SEG_SAMPLE_SIZE * 3))
        self.K = k
        self.train_dataFile = pd.DataFrame(columns=('class', 'filePath'))
        self.test_dataFile = pd.DataFrame(columns=('class', 'filePath'))
        self.enable_overlap = enable_overlap

    def generateSegments(self, fileName, dir):
        x_out = np.empty((0, self.SEG_SAMPLE_SIZE * 3))
        datafile = pd.read_csv(data_folder + dir + "/" + fileName, sep=" ", header=None).values
        length = datafile.shape[0]

        if self.enable_overlap:
            total_segments = int((length - self.SEG_SAMPLE_SIZE) / self.SEG_OVERLAP)
            for segments in range(total_segments):
                x_out = np.append(x_out, [
                    datafile[segments * self.SEG_OVERLAP: segments * self.SEG_OVERLAP + self.SEG_SAMPLE_SIZE, ].reshape(self.SEG_SAMPLE_SIZE * 3)],
                                  axis=0)
        else:
            total_segments = int(length / self.SEG_SAMPLE_SIZE)
            for segments in range(total_segments):
                x_out = np.append(x_out, [
                    datafile[segments * self.SEG_SAMPLE_SIZE:(segments + 1) * self.SEG_SAMPLE_SIZE, ].reshape(self.SEG_SAMPLE_SIZE * 3)], axis=0)
        return x_out

    def createData(self):
        for index, dir in enumerate(dirs):
            y_labels[index + 1] = dir
            y_index[dir] = index + 1
            files = os.listdir(data_folder + dir)
            sample_size = len(files)
            test_size = round(sample_size * 0.2)
            train_size = sample_size - test_size
            train_files = np.random.choice(range(sample_size), train_size, replace=False)

            total_array = np.arange(sample_size)
            include_idx = set(train_files)
            mask = np.array([(i in include_idx) for i in range(len(total_array))])
            test_files = total_array[~mask]

            for index in train_files:
                self.train_dataFile = self.train_dataFile.append({'class': dir, 'filePath': files[index]}, ignore_index=True)

            for index in test_files:
                self.test_dataFile = self.test_dataFile.append({'class': dir, 'filePath': files[index]}, ignore_index=True)

            for train_idx in train_files:
                self.X_Dict = np.append(self.X_Dict, self.generateSegments(files[train_idx], dir), axis=0)

    def createDictionary(self):
        self.dictionary = KMeans(n_clusters=self.K, random_state=random_state).fit(self.X_Dict)

    def getHistData(self, x):
        y_hat = self.dictionary.predict(x)
        row = np.zeros([self.K])
        for i in range(self.K):
            idx = np.where(y_hat == i)
            row[i] = len(idx[0])
        return row

    def getData(self, dataFile):
        X_data = np.empty((0, self.K))
        y_data = np.empty((0, 1), dtype=np.int8)
        for index, row in dataFile.iterrows():
            seg_data = self.generateSegments(row['filePath'], row['class'])
            hist_data = self.getHistData(seg_data)
            X_data = np.append(X_data, [hist_data], axis=0)
            y_data = np.append(y_data, [y_index[row['class']]])

        return (X_data, y_data)

    def getErrorRate(self):
        X_train, y_train = self.getData(self.train_dataFile)
        X_test, y_test = self.getData(self.test_dataFile)

        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(max_depth=50, random_state=0, n_estimators=200)
        clf.fit(X_train, y_train)
        return (1 - clf.score(X_test, y_test)) * 100


if __name__ == '__main__':
    for seg_size in [32]:
        for K in [60, 120, 240, 480, 960]:
            np.random.seed(0)
            cls = VCClassifier(k=K, segment_size=seg_size, overlap=0, random_seg=0, enable_overlap=False)
            cls.createData()
            cls.createDictionary()
            print("Seg Size=", seg_size, ", K=", K, ", Error Rate=", cls.getErrorRate(), "%")
        print("\n")

    for seg_size in [8, 16, 32, 40]:
        for K in [60, 120, 240, 480, 960]:
            np.random.seed(0)
            cls = VCClassifier(k=K, segment_size=seg_size, overlap=0, random_seg=0, enable_overlap=False)
            cls.createData()
            cls.createDictionary()
            print("Seg Size=", seg_size, ", K=", K, ", Error Rate=", cls.getErrorRate(), "%")
        print("\n")

    for seg_size in [16, 32, 40]:
        for K in [120, 240, 480, 960]:
            for overlap in [4, 8, 12]:
                np.random.seed(0)
                cls = VCClassifier(k=K, segment_size=seg_size, overlap=overlap, random_seg=0, enable_overlap=True)
                cls.createData()
                cls.createDictionary()
                print("Seg Size=", seg_size, ", K=", K, ", Overlap=", overlap, ", Error Rate=", cls.getErrorRate(), "%")
            print("\n")
