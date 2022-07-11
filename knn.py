import pandas as pd
from math import dist
import matplotlib.pyplot as plt

# Tim Le Large
# 1910898

# K-Nearest-Neighbor


class Knn:

    def __init__(self, prototype):
        self.data = self.read_data()
        k = 5
        prototype.append(self.get_class(prototype, k))
        self.view_data(prototype)

    # read spiral data from data.txt
    def read_data(self):
        data = []
        with open(r"data.txt") as file:
            for line in file:
                line = [float(value) for value in line.strip().split(";")]
                data.append(line)
        return data

    # view all spiral datapoints
    def view_data(self, prototype):
        self.data.append(prototype)
        self.df = pd.DataFrame(self.data, columns=["x", "y", "class"])
        plt.scatter(self.df['x'], self.df['y'], c=self.df['class'])
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig("spirale.png",
                    format='png', dpi=150)
        plt.show()

    def get_class(self, prototype, k):
        distances = []
        classes = []
        # get distances
        for point in self.data:
            distances.append(dist(prototype[0:2], point[0:2]))
            classes.append(point[2])
        # get k smallest distances
        end_class = 0
        for i in range(0, k):
            index_min_distances = distances.index(min(distances))

            end_class += classes[index_min_distances]
            # remove smallest datapoint distance
            classes.pop(index_min_distances)
            distances.pop(index_min_distances)
        if(end_class > 0):
            return 1
        else:
            return -1


Knn([0.3, -0.5])
