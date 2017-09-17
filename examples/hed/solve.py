from __future__ import division
import numpy as np
import sys
caffe_root = '../../' 
sys.path.insert(0, caffe_root + 'python')
import caffe
import time
import os
import glob

def get_latest_snapshot_number():
    solver_state_paths = glob.glob('snapshot_iter*.solverstate')
    if len(solver_state_paths) > 0:
        start_number_idx = 'snapshot_iter_@'.find('@')
        return max(map(lambda path: int(path[start_number_idx:path.find('.')]), solver_state_paths))
    else:
        return None

def snapshot_number_to_solverstate_str(snapshot_number):
    return 'snapshot_iter_' + str(snapshot_number) + '.solverstate'

def snapshot_number_to_caffemodel_str(snapshot_number):
    return 'snapshot_iter_' + str(snapshot_number) + '.caffemodel'

# make a bilinear interpolation kernel
# credit @longjon
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k:
            print 'input + output channels need to be the same'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt

def main():
    max_time_seconds = float(os.environ['TIMEOUT_SECONDS'])
    start_time = time.time()
    # base net -- follow the editing model parameters example to make
    # a fully convolutional VGG16 net.
    # http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/net_surgery.ipynb
    base_weights = '5stage-vgg.caffemodel'

    # init
    caffe.set_mode_gpu()
    caffe.set_device(0)

    solver = caffe.SGDSolver('solver.prototxt')

    latest_snapshot_number = get_latest_snapshot_number()
    if latest_snapshot_number is not None:
        solver.restore(snapshot_number_to_solverstate_str(latest_snapshot_number))
        solver.net.copy_from(snapshot_number_to_caffemodel_str(latest_snapshot_number))
    else:
        solver.net.copy_from(base_weights)

    # do net surgery to set the deconvolution weights for bilinear interpolation
    interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
    interp_surgery(solver.net, interp_layers)


    # solve straight through -- a better approach is to define a solving loop to
    # 1. take SGD steps
    # 2. score the model by the test net `solver.test_nets[0]`
    # 3. repeat until satisfied
    max_nsteps = 100000
    nsteps = 0
    step_interval = 75
    solver.test_nets[0].share_with(solver.net)
    while nsteps < max_nsteps and (time.time() - start_time <= max_time_seconds):
        solver.step(step_interval)
        print 'Running forward on val data'
        solver.test_nets[0].forward()
        for i in range(1, 6):
            print 'My dsn', i, 'loss', solver.test_nets[0].blobs['dsn' + str(i) + '_loss'].data
        print 'My fuse_loss', solver.test_nets[0].blobs['fuse_loss'].data
        nsteps += step_interval
        print 'Completed', nsteps, ', elapsed time (s):', time.time() - start_time
    print 'Completed', nsteps
    print 'Step interval', step_interval

if __name__ == '__main__':
    main()
