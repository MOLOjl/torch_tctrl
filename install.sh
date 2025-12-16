conda create -n ae python=3.10
conda activate ae
pip install -r requirements.txt
conda install cmake ninja
conda install mkl mkl-include
conda install -c pytorch magma-cuda110
make triton
pip install typing_extensions
pip install pyyaml
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}s
BUILD_TEST=0 python setup.py develop

pip install -r ./ae_benchmark/requirements.txt
cd ./ae_benchmark/alphafold
bash install_openfold.sh
