import numpy as np
import scipy
import scipy.ndimage

def compute_image_mean(paths_lst):
    rgb_sum = np.array([0.0, 0.0, 0.0])
    for path in paths_lst:
        img = scipy.ndimage.imread(path)
        for channel in range(0, 3):
            rgb_sum[channel] += np.sum(img[:, :, channel])
    return rgb_sum / float(len(paths_lst))


def main(path_to_train_lst):
    paths_lst = open(path_to_train_lst)
    print compute_image_mean(paths_lst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Image Mean')
    parser.add_argument('--train-list', dest='train_lst_path')
    parser.parse_args()
    main(parser.train_lst_path)