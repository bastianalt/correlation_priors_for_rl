#!/bin/bash
version=$(python -c "import platform; print(platform.python_version())")
if [ "$version" != "3.6.8" ]
then
	echo "Requires Python 3.6.8"
	exit 1
fi
pip install --upgrade pip
pip install cython
pip install cvxpy
pip install intel-numpy
pip install intel-scipy
pip install intel-scikit-learn
git clone https://github.com/slinderman/pypolyagamma
cd pypolyagamma
pip install -v -e .
cd ..
pip install -r requirements.txt
