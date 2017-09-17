import Image
import numpy as np
from solve.py import get_latest_snapshot_number, snapshot_number_to_caffemodel_str
# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples/hed/
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

def main():
	data_root = '../../data/'
	with open(data_root + 'val_pair.lst') as f:
	    test_lst = f.readlines()
	    
	#test_lst = [data_root + x.strip() for x in test_lst] # full paths to each training image

	test_lst = test_lst[0:min(len(test_lst, 15))]

	im_lst = []
	for i in range(0, len(test_lst)):
	    im = Image.open(data_root + test_lst[i])
	    in_ = np.array(im, dtype=np.float32)
	    in_ = in_[:,:,::-1]
	    in_ -= np.array((17.84271756, 22.54725679, 36.89356086))
	    im_lst.append(in_)

	#remove the following two lines if testing with cpu
	caffe.set_mode_gpu()
	caffe.set_device(0)

	# load net
	latest_snapshot_number = get_latest_snapshot_number()
	if latest_snapshot_number is None:
		print 'Error no snapshot caffemodel available!'
		return
	net = caffe.Net('deploy.prototxt', snapshot_number_to_caffemodel_str(latest_snapshot_number), caffe.TEST)

	for idx in range(0, len(im_list)):
		in_ = im_lst[idx]
		in_ = in_.transpose((2,0,1))
		# shape for input (data blob is N x C x H x W), set data
		net.blobs['data'].reshape(1, *in_.shape)
		net.blobs['data'].data[...] = in_
		# run net and take argmax for prediction
		net.forward()
		out1 = net.blobs['sigmoid-dsn1'].data[0][0,:,:]
		out2 = net.blobs['sigmoid-dsn2'].data[0][0,:,:]
		out3 = net.blobs['sigmoid-dsn3'].data[0][0,:,:]
		out4 = net.blobs['sigmoid-dsn4'].data[0][0,:,:]
		out5 = net.blobs['sigmoid-dsn5'].data[0][0,:,:]
		fuse = net.blobs['sigmoid-fuse'].data[0][0,:,:]

		img_number_str = test_lst[idx][(test_lst[idx].rfind('/') + 1):] # e.g. "0.0.png"
		fuse.save('fuse_output_' + img_number_str)

if __name__ == '__main__':
	main()
