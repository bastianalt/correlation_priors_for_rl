# create and activate virtual environment
~/.pyenv/versions/3.6.8/bin/python -m venv venv/default
source venv/default/bin/activate

# basic setup
pip install --upgrade pip
pip install cython

# install default scientific libraries
pip install numpy==1.16
pip install scipy==1.1.0
pip install scikit-learn==0.20.1

# install requirements
pip install -r requirements.txt
