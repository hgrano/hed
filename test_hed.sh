module purge
module load Python/2.7.13-foss-2016b
mkdir -p $FASTDIR/virtualenvs/hed
source $FASTDIR/virtualenvs/hed/bin/activate # activate env
pip install Cython
pip install numpy
pip install scipy
pip install scikit-image
pip install matplotlib
pip install ipython
pip install h5py
pip install leveldb
pip install networkx
pip install nose
pip install pandas
pip install python-dateutil>=1.4,<2
pip install protobuf
pip install python-gflags
pip install pyyaml
pip install Pillow
pip install six
pip install pypng

# hed requirements
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

export PYTHONPATH=$FASTDIR/hed/python:$PYTHONPATH # change as necessary to your hed
export LD_LIBRARY_PATH=$FASTDIR/boost_py/lib:$LD_LIBRARY_PATH # change path as necessary to your installation of Boost.Python

cd examples/hed
python compute_segmentation.py --data-root ../../real_data --pair-lst real.lst
