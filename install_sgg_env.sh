conda create --name recover_sg python=3.7 --yes
eval "$(conda shell.bash hook)"
conda activate recover_sg
conda install ipython scipy h5py --yes
pip install ninja yacs cython matplotlib tqdm overrides tensorboard ipykernel networkx nltk
python -m ipykernel install --user --name=recover_sg
conda config --add channels nvidia
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge --yes
conda install -c menpo opencv --yes

export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

cd $INSTALL_DIR
python setup.py build develop

unset INSTALL_DIR
