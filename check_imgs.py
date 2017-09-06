def check_imgs(paths_lst):
    for line in paths_lst:
 		split = line.split()
   		    for p in range(0, 2):
	            path = split[p]
                img = scipy.ndimage.imread(path)
                shape = img.shape
                if len(shape) != 3:
                    print 'Warning', path, 'is not RGB'
                else:
                    I, J, K = shape
                    for i in range(0, I):
                        for j in range(0, J):
                            for k in range(0, K):
                                pixel = img[i][j][k]
                                if p == 1 and not (pixel == 0 or pixel == 255):
                                    print 'Warning', path, '[', i, j, k, '] is gt and =', pixel
                                if np.isnan(img[i, j, k]):
                                    print 'Warning', path, '[', i, j, k, '] is NaN'
