import numpy as np 
import matplotlib.pyplot as plt
from collections import Counter

X = np.array([[2, 3,7],
              [3, 3,1],
              [3, 4,3],
              [5, 4,5],
              [5, 6,2],
              [6, 5,3],
              [8, 5,8],
              [8, 6,4]])

y = np.array(['blue', 'blue', 'blue', 'blue', 'red', 'red', 'red', 'red'])

X_test = np.array([[2, 3,7],
                   [7, 5,6],
                   [1, 5, 9]])


class Knn:
    def __init__(self, k):
        self.k = k

    def getDistance(self, p, q):
        return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_test):
        y_pred = []
        
        for new_point in X_test:
            distances = []
            for i, point in enumerate(self.X):
                distance = self.getDistance(point, new_point)
                distances.append([distance, self.y[i]])
            
            categories = [category[1] for category in sorted(distances)[:self.k]]
            result = Counter(categories).most_common(1)[0][0]
            y_pred.append(result)
        
        return y_pred


        
    def visualize(self, X_test):
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(projection="3d")
        ax.grid(True, color="#323232")
        ax.set_facecolor("gray")
        ax.figure.set_facecolor("#121212")
        ax.tick_params(axis="x", color="white")
        ax.tick_params(axis="y", color="white")

        for i, point in enumerate(self.X):
            if self.y[i] == "blue":
                clr = "blue"
            else:
                clr = "red"
            ax.scatter(point[0], point[1], point[2], color=clr, s=60)

        for new_point in X_test:
            check = self.predict(new_point)
            clr = "#FF0000" if check == "red" else "#104DCA"
            ax.scatter(new_point[0], new_point[1], new_point[2], color=clr, marker="*", s=200, zorder=100)

            for c, point in enumerate(self.X):
                if y[c] == "blue":
                   clr ="blue"
                else:
                   clr ="red"
                ax.plot([new_point[0], point[0]], [new_point[1], point[1]], [new_point[2], point[2]], color=clr, linestyle="--", linewidth=1)

        plt.show()



clf = Knn(3)
clf.fit(X,y)
print(clf.predict(X_test))
clf.visualize(X_test)
y_pred = clf.predict(X_test)
print("Tahmin edilen sınıflar:", y_pred)


