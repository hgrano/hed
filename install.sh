make clean
module purge

# Modules needed for install of HED
module load Python/2.7.13-foss-2016b
module load Cython/0.25.2-foss-2016b-Python-2.7.13
module load numpy/1.11.1-foss-2016b-Python-2.7.13
module load scipy/0.17.1-foss-2016b-Python-2.7.13
module load h5py/2.6.0-foss-2016uofa-Python-2.7.12-HDF5-1.8.18
module load nose-parameterized/0.5.0-foss-2016uofa-Python-2.7.12
module load pandas/0.19.1-foss-2015b-Python-2.7.11
module load gflags/2.1.2-foss-2016uofa # is this the python binding?
module load PyYAML/3.12-foss-2016uofa-Python-2.7.12
module load Pillow/3.2.0-foss-2015b-Python-2.7.11 # may cause error

# We don't have: 
# scikit-image>=0.9.3
# matplotlib>=1.3.1
# ipython>=3.0.0
# leveldb>=0.191 (python bindings?)
# networkx>=1.8.1
# python-dateutil>=1.4,<2
# protobuf>=2.5.0 (python bindings?)
# six>=1.1.0


module load Boost/1.61.0-foss-2016b
module load protobuf/2.6.1-foss-2016b
module load glog/0.3.3-foss-2016b
module load gflags/2.1.2-foss-2016b
module load HDF5/1.8.17-foss-2016b-serial
module load OpenCV/3.1.0-foss-2016b
module load LMDB/0.9.18-foss-2016b
module load LevelDB/1.18-foss-2016b
module load snappy/1.1.3-foss-2016b
module load CUDA/8.0.61

make all -j32
make pycaffe -j32

# Module needed for running HED training

export PYTHONPATH=$FASTDIR/hed/python:$PYTHONPATH