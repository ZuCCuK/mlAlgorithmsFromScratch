import numpy as np 
import matplotlib.pyplot as plt
from collections import Counter

points = {"blue": [[2, 4], [1, 3], [2, 3], [3, 2], [2, 1]],
          "red": [[5, 4], [5, 6], [4, 5], [4, 6], [6, 6]]}
new_point = [3, 4]

class Knn:
    def __init__(self, k):
        self.k = k

    def getDistance(self, p, q):
        return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))

    def fit(self, points):
        self.points = points

    def predict(self, new_point):
        distances = []

        for category in self.points:
            for point in self.points[category]:
                distance = self.getDistance(point, new_point)
                distances.append([distance, category])
        categories = [category[1] for category in sorted(distances)[:self.k]]
        result = Counter(categories).most_common(1)[0][0]
        return result

clf = Knn(3)
clf.fit(points)
print(clf.predict(new_point))

ax = plt.subplot()
ax.grid(True, color="#323232")
ax.figure.set_facecolor("#121212")
ax.tick_params(axis="x", color="white")
ax.tick_params(axis="y", color="white")

for point in points["blue"]:
    ax.scatter(point[0], point[1], color="#104DCA", s=60)

for point in points["red"]:
    ax.scatter(point[0], point[1], color="#FF0000", s=60)

check = clf.predict(new_point)
color = "#FF0000" if check == "red" else "#104DCA"

ax.scatter(new_point[0], new_point[1], color=color, marker="*", s=200, zorder=100)

for point in points["blue"]:
    ax.plot([new_point[0], point[0]], [new_point[1], point[1]], color="#104DCA", linestyle="--", linewidth=1)

for point in points["red"]:
    ax.plot([new_point[0], point[0]], [new_point[1], point[1]], color="#FF0000", linestyle="--", linewidth=1)

plt.show()
