make clean
module purge

module load Boost/1.61.0-foss-2016uofa-Python-2.7.11
# Python setup for HED
module load Python/2.7.11-foss-2016uofa

module load protobuf/2.6.1-foss-2016b
module load glog/0.3.3-foss-2016b
module load gflags/2.1.2-foss-2016b
module load HDF5/1.8.17-foss-2016b-serial
module load OpenCV/3.1.0-foss-2016b
module load LMDB/0.9.18-foss-2016b
module load LevelDB/1.18-foss-2016b
module load snappy/1.1.3-foss-2016b
module load CUDA/8.0.61


mkdir -p $FASTDIR/virtualenvs/hed
source $FASTDIR/virtualenvs/hed/bin/activate # activate env
for req in $(cat python/requirements.txt); do pip install $req; done

# C/C++ modules for HED compilation
# module load Boost/1.61.0-foss-2016b # or Boost/1.61.0-foss-2016uofa-Python-2.7.11
# module load protobuf/2.6.1-foss-2016b
# module load glog/0.3.3-foss-2016b
# module load gflags/2.1.2-foss-2016b
# module load HDF5/1.8.17-foss-2016b-serial
# module load OpenCV/3.1.0-foss-2016b
# module load LMDB/0.9.18-foss-2016b
# module load LevelDB/1.18-foss-2016b
# module load snappy/1.1.3-foss-2016b
# module load CUDA/8.0.61

# pip install Cython
# pip install numpy
# pip install scipy
# pip install scikit-image
# pip install matplotlib
# pip install ipython
# pip install h5py
# pip install leveldb
# pip install networkx
# pip install nose
# pip install pandas
# pip install python-dateutil>=1.4,<2
# pip install protobuf
# pip install python-gflags
# pip install pyyaml
# pip install Pillow
# pip install six

# make all -j32
# make pycaffe -j32

# Module needed for running HED training

# export PYTHONPATH=$FASTDIR/hed/python:$PYTHONPATH