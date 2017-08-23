module purge

# Modules needed for install of HED
module load Python/2.7.13-foss-2016b
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

# Module needed for running HED training
module load numpy/1.11.1-foss-2016b-Python-2.7.13

export PYTHONPATH=$FASTDIR/hed/python:$PYTHONPATH