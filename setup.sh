brew reinstall gcc
brew install gsl
CC=$(which gcc-8) CXX=$(which g++-8) pip install -r requirements.txt
