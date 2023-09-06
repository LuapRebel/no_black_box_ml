import csv
from sklearn.neighbors import KNeighborsClassifier

class_list = ["car", "fish", "house", "tree", "bicycle", "guitar", "pencil", "clock"]
classes = {k: v for v, k in enumerate(class_list)}


def readFeatureFile(filePath):
    X = []
    y = []
    with open(filePath, "r") as csvfile:
        lines = csv.reader(csvfile)
        next(lines, None)
        for row in lines:
            X.append([float(row[0]), float(row[1])])
            y.append(row[2])
    return (X, y)


X, y = readFeatureFile("data/dataset/training.csv")
knn = KNeighborsClassifier(n_neighbors=50, algorithm="brute", weights="uniform")

knn.fit(X, y)

X, y = readFeatureFile("data/dataset/testing.csv")

accuracy = knn.score(X, y)
print(f"Accuracy: {accuracy}")
