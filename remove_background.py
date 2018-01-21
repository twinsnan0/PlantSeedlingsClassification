from scipy.stats import multivariate_normal
import numpy as np
import cv2
import matplotlib.pyplot as plt

LEAF_COLOR_MU = [67.0507, 82.7032, 31.5556]
LEAF_COLOR_SIGMA = [[146.4852, 146.0210, 34.7797],
                    [146.0210, 160.2452, 30.7043],
                    [34.7797, 30.7043, 143.9793]]
THRESHOLD = 0.00000005


def remove_background(image):
    # probability of each pixel being on the plant
    # based on multivariate normal distribution
    prob = multivariate_normal.pdf(image, mean=LEAF_COLOR_MU, cov=LEAF_COLOR_SIGMA)
    mask = prob > THRESHOLD
    mask = np.repeat(np.expand_dims(mask, axis=2), 3, axis=2)
    image = np.multiply(image, mask)
    return image


if __name__ == "__main__":
    img = cv2.imread('test.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = remove_background(img)

    plt.imshow(img)
    plt.show()

# edges = cv2.Canny(img, 0, 200)
# plt.imshow(edges)
# plt.show()
