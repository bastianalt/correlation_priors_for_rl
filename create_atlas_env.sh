# create and activate virtual environment
~/.pyenv/versions/3.6.8/bin/python -m venv venv/atlas
source venv/atlas/bin/activate

# basic setup
pip install --upgrade pip
pip install cython==0.29.3

# compile default scientific libraries with atlas
cd venv/atlas
path=`pwd`
mkdir download
pip download --no-binary :all: numpy -d download/numpy
pip download --no-binary :all: scipy -d download/scipy
pip download --no-binary :all: scikit-learn -d download/scikit-learn
unzip download/numpy/*.zip -d build/
mv build/numpy* build/numpy
tar -xvzf download/scipy/*.gz -C build/
mv build/scipy* build/scipy
tar -xvzf download/scikit-learn/*.gz -d build/
mv build/scikit-learn* build/scikit-learn
rm -R download
cp ../../site.cfg build/numpy
cp ../../site.cfg build/scipy
cp ../../site.cfg build/scikit-learn
export PYTHONPATH=${path}/lib64/python3.6/site-packages
cd build/numpy
python setup.py install --prefix ${path}
cd ../scipy
python setup.py install --prefix ${path}
cd ../scikit-learn
python setup.py install --prefix ${path}
cd ../../../..

# install requirements
pip install -r requirements.txt
