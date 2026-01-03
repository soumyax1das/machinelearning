import pandas as pd
import numpy as np


def calculate_distance(p1, p2):
    """
    Calculate Euclidean distance between two points (arrays or lists)
    """
    return np.sqrt(np.sum((p1 - p2) ** 2))


if __name__ == '__main__':

    # Step 1: Read CSV
    df = pd.read_csv("/Users/soumya/Documents/coding/python/machinelearning/dataset/simple_clustering_input.csv")
    print("Initial Data:")
    print(df.head())

    # Step 2: Choose number of clusters (k)
    k = 3

    # Step 3: Randomly select k points/samples as initial centroids
    # It also maintains the index column in the df. So the index column is removed from the df
    # through reset_index(drop=True)
    cur_centroids = df.sample(n=k, random_state=42).reset_index(drop=True)
    print("\nInitial Centroids:")
    print(cur_centroids)

    # Step 4: Repeat until centroids stabilize
    while True:
        # Assign each point to the nearest centroid
        distances = []
        for i in range(k):
            dist = np.sqrt(((df[['age', 'income']] - cur_centroids.iloc[i]) ** 2).sum(axis=1))
            distances.append(dist)

        df['cluster'] = np.argmin(distances, axis=0)

        # Calculate new centroids as mean of points in each cluster
        new_centroids = df.groupby('cluster')[['age', 'income']].mean().reset_index(drop=True)

        # Check for convergence (if centroids didn't change)
        if new_centroids.equals(cur_centroids):
            print("\nCentroids stabilized — clustering complete.")
            break
        else:
            cur_centroids = new_centroids

    # Step 5: Show results
    print("\nFinal Centroids:")
    print(cur_centroids)

    print("\nClustered Data Sample:")
    print(df.head(10))
