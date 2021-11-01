import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.sparse
import warnings
import cv2

TEST_K = [1, 5, 10, 15, 20]

IMG_DIR = '../img'
OUT_DIR = '../result'


def knn_matting(K, img, trimap, my_lambda=100):
    [h, w, c] = img.shape
    img, trimap = img / 255.0, trimap / 255.0
    foreground = (trimap == 1.0).astype(int)
    background = (trimap == 0.0).astype(int)
    all_constraints = foreground + background
    img_size = h * w

    # Calculate feature vector X
    # Hint: X(i) = (R, G, B, x, y)(i)
    rgb = img.reshape(img_size, c)
    x, y = np.unravel_index(np.arange(img_size), (h, w))
    xy = np.array([x, y]).T / np.sqrt(h * h + w * w)
    X = np.append(rgb, xy, axis=1)

    # Calculate nearest neighbors
    nn = NearestNeighbors(n_neighbors=K, n_jobs=4).fit(X)
    knn = nn.kneighbors(X)[1]

    # Calculate kernel function k
    # Hint: k(i, j) = 1 - |X(i) - X(j)| / C
    i = np.repeat(np.arange(img_size), K)
    j = knn.reshape(img_size * K)
    C = c + 2
    k = 1 - np.linalg.norm(X[i] - X[j], axis=1) / C

    # Calculate the affinity matrix A
    A = scipy.sparse.coo_matrix((k, (i, j)), shape=(img_size, img_size))

    # Prepare for objective function
    # Hint: obj = (L + lambda * M) * alpha - lambda * v
    D = scipy.sparse.diags(np.ravel(A.sum(axis=1)))
    L = D - A
    M = scipy.sparse.diags(np.ravel(all_constraints[:, :, 0]))
    v = np.ravel(foreground[:, :, 0]).T

    # Solve for the linear system
    warnings.filterwarnings('error')
    alpha = []
    try:
        alpha = np.minimum(np.maximum(
            scipy.sparse.linalg.spsolve(
                L + my_lambda * M,
                my_lambda * v,
            ), 0), 1).reshape(h, w)
    except Warning:
        tmp = scipy.sparse.linalg.lsqr(
            L + my_lambda * M,
            my_lambda * v,
        )
        alpha = np.minimum(np.maximum(tmp[0], 0), 1).reshape(h, w)

    return alpha


def main():
    for K in TEST_K:
        img_names = ['bear.png', 'gandalf.png', 'woman.png'] + [f'GT{i:02}.png' for i in range(1, 28)]
        print(f'K = {K}.')

        for img_name in img_names:
            print(f'Processing image {img_name}.')

            image = cv2.imread(f'{IMG_DIR}/image/{img_name}')
            trimap = cv2.imread(f'{IMG_DIR}/trimap/{img_name}')

            alpha = knn_matting(K, image, trimap)
            alpha = np.stack((alpha,) * 3, axis=-1)

            background = cv2.imread(f'{IMG_DIR}/background/bg_{img_name}')
            if background is None:
                background = cv2.imread(f'{IMG_DIR}/background/bg_general.png')
            background = cv2.resize(background, (image.shape[1], image.shape[0]))

            compose = alpha * image + (1 - alpha) * background
            compose = compose.astype(image.dtype)

            cv2.imwrite(f'{OUT_DIR}/{K}_{img_name}', compose)
            print('Done.')


if __name__ == '__main__':
    main()
