# Script for setting up AWS EC2 machine
# (run Ubuntu on g2.2xlarge or p2.xlarge instance type)

# Base Setup
wget https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64-deb
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64-deb
sudo apt-get update
sudo apt-get install -y cuda python3-pip python3-numpy python3-dev python3-wheel
rm cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64-deb

# CUDNN Setup
tar xvzf ~/cudnn-8.0-linux-x64-v5.1.tgz
sudo cp -P ~/cuda/include/cudnn.h /usr/local/cuda/include
sudo cp -P ~/cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
rm ~/cudnn-8.0-linux-x64-v5.1.tgz

# Python Setup
pip3 install --upgrade pip
sudo pip3 install --upgrade 'https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0rc2-cp35-cp35m-linux_x86_64.whl'
sudo pip3 install pycrypto
sudo pip3 install h5py
sudo pip3 install theano
sudo pip3 install keras

# Environment Setup
echo 'export PATH=/usr/local/cuda-8.0/bin:$PATH' >> ~/.bash_profile
git clone https://github.com/stevekrenzel/modeled_encryption

# Cleanup
rm setup.sh
