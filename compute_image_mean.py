import numpy as np
import scipy
import scipy.ndimage
import argparse

def compute_image_mean(paths_lst):
    rgb_sum = np.array([0.0, 0.0, 0.0])
    num_pixels = 0
    for line in paths_lst:
        path = line.split()[0]
        img = scipy.ndimage.imread(path)
        shape = img.shape
        if len(shape) != 3:
            print 'Warning', path, 'is not RGB'
        else:
            num_pixels += shape[0] * shape[1]
            for channel in range(0, 3):
                rgb_sum[channel] += np.sum(img[:, :, channel])
    return rgb_sum / float(num_pixels)


def main(path_to_train_lst):
    paths_lst = open(path_to_train_lst)
    print compute_image_mean(paths_lst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Image Mean')
    parser.add_argument('--train-list', dest='train_lst_path')
    args = parser.parse_args()
    main(args.train_lst_path)
