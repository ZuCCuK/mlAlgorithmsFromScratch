import numpy as np 
import matplotlib.pyplot as plt
from collections import Counter

points = {"blue": [[1,2, 4], [1,2, 3], [3,2, 3], [3, 2,4], [2, 3,1]],
          "red": [[5, 4,6], [9,5, 6], [10,4, 5], [8,4, 6], [5,6, 6]]}
new_point = [3, 4, 5]

class Knn:
    def __init__(self, k):
        self.k = k

    def getDistance(self, p, q):
        return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))

    def fit(self, points):
        self.points = points

    def predict(self, new_point):
        distances = []
        self.new_point = new_point

        for category in self.points:
            for point in self.points[category]:
                distance = self.getDistance(point, new_point)
                distances.append([distance, category])
        categories = [category[1] for category in sorted(distances)[:self.k]]
        result = Counter(categories).most_common(1)[0][0]
        return result
    
    def visualize(self):
        fig = plt.figure(figsize=(15,12))
        ax = fig.add_subplot(projection="3d")
        ax.grid(True, color="#323232")
        ax.set_facecolor("black")
        ax.figure.set_facecolor("#121212")
        ax.tick_params(axis="x", color="white")
        ax.tick_params(axis="y", color="white")

        for point in self.points["blue"]:
            ax.scatter(point[0], point[1],point[2], color="#104DCA", s=60)#buraya useniom yarın yapcm

        for point in self.points["red"]:
            ax.scatter(point[0], point[1],point[2], color="#FF0000", s=60)

        check = clf.predict(new_point)
        color = "#FF0000" if check == "red" else "#104DCA"

        ax.scatter(self.new_point[0], self.new_point[1],self.new_point[2], color=color, marker="*", s=200, zorder=100)

        for point in points["blue"]:
            ax.plot([self.new_point[0], point[0]], [self.new_point[1], point[1]],[self.new_point[2], point[2]], color="#104DCA", linestyle="--", linewidth=1)

        for point in points["red"]:
            ax.plot([self.new_point[0], point[0]], [self.new_point[1],point[1]],[self.new_point[2], point[2]], color="#FF0000", linestyle="--", linewidth=1)

        plt.show()


clf = Knn(3)
clf.fit(points)
print(clf.predict(new_point))
clf.visualize()



