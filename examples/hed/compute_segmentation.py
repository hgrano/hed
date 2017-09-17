from PIL import Image
import numpy as np
import png
# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples/hed/
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
from solve import get_latest_snapshot_number, snapshot_number_to_caffemodel_str
import argparse

def main(caffe_mode):
	data_root = '../../data/'
	with open(data_root + 'val_pair.lst') as f:
	    test_lst = [x.split()[0] for x in f.readlines()] # take first item - the image
	    
	#test_lst = [data_root + x.strip() for x in test_lst] # full paths to each training image

	test_lst = test_lst[0:min(len(test_lst), 15)]

	im_lst = []
	for i in range(0, len(test_lst)):
	    im = Image.open(data_root + test_lst[i])
	    in_ = np.array(im, dtype=np.float32)
	    in_ = in_[:,:,::-1]
	    in_ -= np.array((17.84271756, 22.54725679, 36.89356086))
	    im_lst.append(in_)

	#remove the following two lines if testing with cpu
	if caffe_mode == 'GPU':
		caffe.set_mode_gpu()
		caffe.set_device(0)
	elif caffe_mode == 'CPU':
		caffe.set_mode_cpu()
	else:
		print 'Unrecognized caffe mode "', caffe_mode, '" Valid options are "CPU" or "GPU"'
		return

	# load net
	latest_snapshot_number = get_latest_snapshot_number()
	if latest_snapshot_number is None:
		print 'Error no snapshot caffemodel available!'
		return
	net = caffe.Net('deploy.prototxt', snapshot_number_to_caffemodel_str(latest_snapshot_number), caffe.TEST)

	for idx in range(0, len(im_lst)):
		in_ = im_lst[idx]
		in_ = in_.transpose((2,0,1))
		# shape for input (data blob is N x C x H x W), set data
		net.blobs['data'].reshape(1, *in_.shape)
		net.blobs['data'].data[...] = in_
		# run net and take argmax for prediction
		net.forward()
		outs = [net.blobs['sigmoid-dsn' + str(i)].data[0][0,:,:] for i in range(1, 6)]
# 		out1 = net.blobs['sigmoid-dsn1'].data[0][0,:,:]
# 		out2 = net.blobs['sigmoid-dsn2'].data[0][0,:,:]
# 		out3 = net.blobs['sigmoid-dsn3'].data[0][0,:,:]
# 		out4 = net.blobs['sigmoid-dsn4'].data[0][0,:,:]
# 		out5 = net.blobs['sigmoid-dsn5'].data[0][0,:,:]
		for i, out in enumerate(outs):
			print 'np.sum(outs[' + str(i) + '] =', np.sum(out)
		fuse = net.blobs['sigmoid-fuse'].data[0][0,:,:]
		print 'np.sum(fuse) =', np.sum(fuse)
		fuse_uint16 = fuse.astype(np.uint16)
		print 'np.sum(fuse.astype(uint16)) =', np.sum(fuse_uint16)
		img_number_str = test_lst[idx][(test_lst[idx].rfind('/') + 1):] # e.g. "0.0.png"
		fuse_uint8 = np.zeros(shape=fuse.shape, dtype=np.uint8)
		rows, cols = fuse.shape
		for i in range(0, rows):
			for j in range(0, cols):
				if fuse[i, j] > 0.0:
					fuse_uint16[i, j] = 255
				elif fuse[i, j] < 0.0:
					print 'hm negative value here...'	
		print 'fuse.shape ==', fuse.shape
		print 'np.sum(fuse_uint8) ==', np.sum(fuse_uint8)
		png.from_array(fuse_uint8, 'L').save('fuse_output_' + img_number_str)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Test out hed + save some images!')
	parser.add_argument('--caffe-mode', dest='caffe_mode', type=str,
                    help='Caffe mode (CPU or GPU)', default='GPU')
	args = parser.parse_args()
	main(args.caffe_mode)
