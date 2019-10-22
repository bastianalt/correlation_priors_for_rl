# create and activate virtual environment
~/.pyenv/versions/3.6.8/bin/python -m venv venv/intel
source venv/intel/bin/activate

# basic setup
pip install --upgrade pip
pip install cython==0.29.3

# install intel scientific libraries
pip install intel-numpy==1.15.1
pip install intel-scipy==1.1.0
pip install intel-scikit-learn==0.19.2

# install requirements
pip install -r requirements.txt
