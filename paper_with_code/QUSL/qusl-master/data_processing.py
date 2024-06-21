import random
import numpy as np
import pandas as pd
from PIL import Image
from os import listdir
import genetic_algorithms
from os.path import isfile, join
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time
from concurrent.futures import ThreadPoolExecutor


def generate_landscape_triplets(base_path, dataset, num_triplets, testing=False):
    path = f'{base_path}//DATASET/{dataset}/{dataset}s_train'
    path_test = f'{base_path}//DATASET/{dataset}/{dataset}s_test'
    image_triplets = []
    image_indecies = []
    if testing:
        files = [f for f in listdir(path_test) if isfile(join(path_test, f))]
        files.sort()
        num_files = len(files)
        # new_width, new_height = 26, 26
        new_width, new_height = 50, 50
        for _ in range(num_triplets):
            indecies = []
            triplets = []
            images = []
            for i in range(3):
                index = int(np.floor(random.random() * num_files))
                im = Image.open(f'{base_path}//DATASET/{dataset}/{dataset}s_test/' + str(files[index]))

                width, height = im.size
                left = (width - new_width) // 2
                top = (height - new_height) // 2
                right = (width + new_width) // 2
                bottom = (height + new_height) // 2
                im = im.crop((left, top, right, bottom))
                r_bin, g_bin, b_bin = get_color_density(np.array(im))

                triplets.append(np.array(r_bin + g_bin + b_bin))
                indecies.append(index)

                images.append(np.reshape(np.array(im), [-1, ]))

            noisy_image = add_noise(np.array(images[0]), noise_type='gaussian', seed=42)

            image_triplets.append((images[0], noisy_image, images[2]))
            image_indecies.append((indecies[0], indecies[0], indecies[2]))

    else:
        files = [f for f in listdir(path) if isfile(join(path, f))]
        files.sort()
        num_files = len(files)
        new_width, new_height = 50, 50

        for _ in range(num_triplets):
            indecies = []
            triplets = []
            images = []
            for i in range(3):
                index = int(np.floor(random.random() * num_files))
                im = Image.open(f'{base_path}/DATASET/{dataset}/{dataset}s_train/' + str(files[index]))


                width, height = im.size
                left = (width - new_width) // 2
                top = (height - new_height) // 2
                right = (width + new_width) // 2
                bottom = (height + new_height) // 2
                im = im.crop((left, top, right, bottom))
                r_bin, g_bin, b_bin = get_color_density(np.array(im))
                triplets.append(np.array(r_bin + g_bin + b_bin))
                indecies.append(index)
                images.append(np.reshape(np.array(im), [-1, ]))

            noisy_image = add_noise(np.array(images[0]), noise_type='gaussian', seed=42)
            image_triplets.append((images[0], noisy_image, images[2]))
            image_indecies.append((indecies[0], indecies[0], indecies[2]))
    return image_triplets, image_indecies


def add_noise(image, noise_type='gaussian', seed=None):
    """
    Add noise to an image.

    Parameters:
        image (numpy.ndarray): Input image.
        noise_type (str): Type of noise ('gaussian', 'salt_and_pepper', 'poisson').
        seed (int): Seed for random number generation.

    Returns:
        numpy.ndarray: Image with added noise.
    """
    if seed is not None:
        np.random.seed(seed)

    noisy_image = image.copy()

    if noise_type == 'gaussian':
        mean = 0
        std = 5  # Adjust the standard deviation based on your preference
        noise = np.random.normal(mean, std, image.shape)
        noisy_image = np.clip(image + noise, 0, 255)

    elif noise_type == 'salt_and_pepper':
        prob = 0.05  # Adjust the probability based on your preference
        mask = np.random.rand(*image.shape) < prob
        noisy_image[mask] = 0 if np.random.rand() < 0.5 else 255

    elif noise_type == 'poisson':
        noisy_image = np.random.poisson(image)
    return noisy_image.astype(np.uint8)


def perform_pca(x, pca_dims=4096):
    """
    Performs PCA on the given training example data with the specified # of dimensions. L2 normalizes data for
    PCA fit and transformation, and returns resulting features scaled to [0, 1] range.
    :param x: Numpy array containing rows of image data training examples.
    :param pca_dims: the number of dimensions (features) to reduce each example to via PCA.
    :return: Numpy array with [0, 1] scaled result of PCA dimensionality reduction.
    """
    pca = PCA(pca_dims)
    # Normalize image data so its pythagorean sum is 1
    pca.fit(preprocessing.normalize(x))
    return pca.transform(preprocessing.normalize(x))


def get_color_density(image):
    num_bins = 8
    bin_threshold = 256 // num_bins
    red = image[..., 0]
    green = image[..., 1]
    blue = image[..., 2]
    red_bin = np.reshape(red // bin_threshold, [-1,])
    green_bin = np.reshape(green // bin_threshold, [-1])
    blue_bin = np.reshape(blue // bin_threshold, [-1])
    red_count = [0 for _ in range(num_bins)]
    blue_count = [0 for _ in range(num_bins)]
    green_count = [0 for _ in range(num_bins)]
    for r, g, b, in zip(red_bin, blue_bin, green_bin):
        red_count[r] += 1
        blue_count[g] += 1
        green_count[b] += 1
    return red_count, blue_count, green_count


def crop_and_resize_image(image):
    new_width, new_height = 80, 80
    width, height = image.size
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    return image.crop((left, top, right, bottom))


def process_feature(l, triplets, indecies, evolution, circuit, number_of_qubits, path, files):
    dists = []
    loss_tuples = []
    distance_tuples = []
    min_loss = 10e5
    closest_dist = 10e5
    closest_index = 0
    anchor_image = triplets[l][0]
    anchor_index = indecies[l][0]
    all_losses = []
    all_distances = []
    for image, indexes in zip(triplets, indecies):
        if indexes[0] != anchor_index:
            flattened = [i for pair in zip(anchor_image, image[0]) for i in pair]
            z_out1 = evolution.runcircuit(circuit, flattened, number_of_qubits=number_of_qubits)
            flattened = [j for pair in zip(anchor_image, image[0]) for j in pair]
            z_out2 = evolution.runcircuit(circuit, flattened, number_of_qubits=number_of_qubits)
            loss = sum([np.abs(i - j) for i, j in zip(z_out1, z_out2)])
            print('loss', loss)

            im1 = Image.open(join(path, str(files[anchor_index])))
            im1 = crop_and_resize_image(im1)
            im2 = Image.open(join(path, str(files[indexes[0]])))
            im2 = crop_and_resize_image(im2)

            anchor_r, anchor_g, anchor_b = get_color_density(np.array(im1))
            new_r, new_g, new_b = get_color_density(np.array(im2))
            dist = np.sum(np.power(np.array(anchor_r + anchor_g + anchor_b) - np.array(new_r + new_g + new_b), 2)) ** 0.5
            dists.append(dist)
            distance_tuples.append((float(dist), files[indexes[0]]))
            loss_tuples.append((float(loss), files[indexes[0]]))

            if loss < min_loss:
                min_loss = loss
                closest_index = indexes[0]
                closest_dist = dist
    print('结束一轮')
    print('anchor_index', files[anchor_index])
    print('indexes', files[closest_index])
    print('closest_dist', closest_dist)

    all_losses.append(loss_tuples)
    all_distances.append(distance_tuples)


    # 将数据转为 DataFrame
    loss_df = pd.DataFrame(loss_tuples, columns=['Loss', 'File'])
    distance_df = pd.DataFrame(distance_tuples, columns=['Distance', 'File'])

    # 保存为 CSV 文件
    loss_df.to_csv(f'D:\pycharm\projects\mindquantun_RGB/result/csv/loss{l}.csv', index=False)
    distance_df.to_csv(f'D:\pycharm\projects\mindquantun_RGB/result/csv/distance{l}.csv', index=False)

    im1 = Image.open(join(path, str(files[anchor_index])))
    im1 = crop_and_resize_image(im1)
    im2 = Image.open(join(path, str(files[closest_index])))
    im2 = crop_and_resize_image(im2)

    Image.fromarray(np.hstack((np.array(im1), np.array(im2)))).save(f'D:\pycharm\projects\mindquantun_RGB/SliqImages/{anchor_index}_{closest_index}.jpg')
    print(f'anchor_index{anchor_index}\nmin_loss{min_loss}\nclosest_index{closest_index}\nclosest_dist{closest_dist}')


def median_spearman_corr(x, y):
    from scipy.stats import spearmanr
    x_rank = x.rank()
    y_rank = y.rank()
    spearman_corr, _ = spearmanr(x_rank, y_rank)
    return spearman_corr

    

def Spearman(base_path, dataset, i, j):
    print(f'###################{i}###################')
    csv_filename = f'{base_path}result/{dataset}/csv/merged_data_{j}_{i}.csv'
    df = pd.read_csv(csv_filename)

    all_losses = df['Loss']
    all_distances = df['Distance']

    plt.scatter(all_losses, all_distances)
    plt.title('Scatter Plot of Losses and Distances')
    plt.xlabel('Loss')
    plt.ylabel('Distance')
    plt.grid(True)

    median_spearman_corr_value = median_spearman_corr(all_losses, all_distances)

    print(f"Median Spearman correlation coefficient: {median_spearman_corr_value}")

    save_path = f'{base_path}result/{dataset}/correlation/plot_{j}_{i}_{median_spearman_corr_value}.png'
    plt.savefig(save_path)

    plt.show()
    print(f'num{j}_{i}individual：{median_spearman_corr_value}')
    return median_spearman_corr_value
