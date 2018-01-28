from scipy.stats import multivariate_normal
import numpy as np
import cv2
import matplotlib.pyplot as plt

LEAF_COLOR_MU = [67.0507, 82.7032, 31.5556]
LEAF_COLOR_SIGMA = [[146.4852, 146.0210, 34.7797],
                    [146.0210, 160.2452, 30.7043],
                    [34.7797, 30.7043, 143.9793]]
THRESHOLD = 0.00000005


def remove_background(images):
    # probability of each pixel being on the plant
    # based on multivariate normal distribution
    if images.shape[1] == 3 and images.ndim == 4:
        images = np.transpose(images, (0, 2, 3, 1))
    prob = multivariate_normal.pdf(images, mean=LEAF_COLOR_MU, cov=LEAF_COLOR_SIGMA)
    if prob.ndim == 2:
        prob = np.expand_dims(prob, axis=0)
    mask = prob > THRESHOLD
    mask_4d = np.expand_dims(mask, axis=3)
    result_images = images * mask_4d
    return prob, mask, result_images


if __name__ == "__main__":
    img = cv2.imread('test.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, _, img = remove_background(img)
    img = np.squeeze(img)
    plt.imshow(img)
    plt.show()

# edges = cv2.Canny(img, 0, 200)
# plt.imshow(edges)
# plt.show()
