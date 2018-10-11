import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

n_colors = 32
pic = 'sutd.png'
img = mpimg.imread(pic)
img = img[:, :, :3]

w, h, d = tuple(img.shape)
image_array = np.reshape(img, (w * h, d))


def recreate_image(palette, labels, w, h):
    d = palette.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0

    for i in range(w):
        for j in range(h):
            image[i][j] = palette[labels[label_idx]]
            label_idx += 1
    return image


# Derive kmeans palette and kmeans labels using k-means clustering.
def givenFigurePlots(kmeans_palette, kmeans_labels, random_palette, random_labels):
    plt.figure(1)
    plt.clf()
    ax = plt.axes([0, 0, 1, 1])
    plt.axis('off')
    plt.title('Original image(16.8 million colors)')
    plt.imshow(img)

    plt.figure(2)
    plt.clf()
    ax = plt.axes([0, 0, 1, 1])
    plt.axis('off')
    plt.title('Compressed image(K - Means)')
    plt.imshow(recreate_image(kmeans_palette, kmeans_labels, w, h))

    plt.figure(3)
    plt.clf()
    ax = plt.axes([0, 0, 1, 1])
    plt.axis('off')
    plt.title('Compressed image(Random)')
    plt.imshow(recreate_image(random_palette, random_labels, w, h))
    plt.show()


def fitModel2a():
    # print("Current image:", image_array.shape)

    image_sample = image_array[rng.randint(w * h, size=1000)]
    kmeans = KMeans(n_clusters=n_colors, random_state=0)
    kmeans = kmeans.fit(image_sample)

    kmeans_palette = kmeans.cluster_centers_
    kmeans_labels = kmeans.predict(image_array)

    # print("pixel labels:\n{}".format(kmeans_labels))

    return kmeans_palette, kmeans_labels


def formRandomPalette2b():
    random_palette = image_array[rng.randint(w * h, size=n_colors)]
    random_labels = pairwise_distances_argmin(X=random_palette,
                                              Y=image_array,
                                              axis=0)
    # print(random_labels)
    return random_palette, random_labels


if __name__ == '__main__':
    kmeans_palette, kmeans_labels = fitModel2a()
    random_palette, random_labels = formRandomPalette2b()
    givenFigurePlots(kmeans_palette, kmeans_labels, random_palette, random_labels)
