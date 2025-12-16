# AdaPipe

## Setup network file system(NFS)

First setting up the NFS for the cluster **if NFS is not provided**.

### NFS server
```
sudo apt update
sudo apt install nfs-kernel-server

# create the directory and change the ownership
sudo mkdir /mnt -p
sudo chown nobody:nogroup /mnt

# add next line to the file `/etc/exports`. (192.168.1.0 is the subnet that can access this directory)
/mnt    192.168.1.0/24(rw,sync,no_root_squash,no_subtree_check)

# apply the configuartion and restart the service
sudo exportfs -a
sudo systemctl restart nfs-kernel-server

# Firewall(If needed)
sudo ufw allow from 192.168.1.0/24 to any port nfs
sudo ufw enable
sudo ufw status
```

### NFS client
```
sudo apt update
sudo apt install nfs-common

sudo mkdir -p /mnt
sudo mount 192.168.1.100:/mnt /mnt
```

## Install software dependencies

Please install following softwares on the network file system.

1. install miniconda
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
```

2. create python environment
```
conda create -n asplos_ae python=3.10
conda activate asplos_ae
```

3. install and activate `spack`
```
git clone https://github.com/spack/spack.git
source spack/share/spack/setup-env.sh
```

4. install and load `cuda`
```
spack install cuda@12.1.0
spack load cuda@12.1.0
```

5. install and load `openmpi`
```
spack install openmpi@5.0.2
spack load openmpi@5.0.2
```

6. install `gcc9`
```
spack install gcc@9.5.0
spack load gcc@9.5.0
```

7. install `PyTorch: 2.2.1+cuda12.1`
```
pip3 install torch torchvision torchaudio
```

8. install `apex`
```
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout b496d85f
pip3 install -r requirements.txt
pip3 install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

9. install python packages for AdaPipe/Megatron-LM
```
cd AdaPipe
pip3 install -r requirements.txt
```

10. setup DP algorithm module
```
cd AdaPipe/recompute_config
g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) adapipe_search.cpp -o adapipe_search$(python3-config --extension-suffix)
```

11. setup the dataset path
modify the variable `DATA_PATH` in `examples/pretrain_single_test.sh` to the path of the enwik8 dataset

12. run the basic test on single node
We provide a script to run a gpt-like model with 8 A100 40GB GPUs connected with PCIe.
```
cd AdaPipe
sh single_gpt_test.sh
```
The iteration time of Even Partitioning and AdaPipe should be around 40.4s and 39.1s.

## Global test

1. set hostfile for mpirun
```
worker0-ip max_slots=8
worker1-ip max_slots=8
```

2. set environment variable of startup scripts

In script `examples/pretrain_gpt_distributed_mpi.sh` and `examples/pretrain_llama2_distributed_mpi.sh`.

```
export MASTER_ADDR=worker0-ip # modify MASTER_ADDR
export DATA_PATH=dataset_path
```

3. run whole test
```
sh global_test.sh hostfile_path
```

4. parse result
```
python3 collect_result.py
```

The expected result is in `expected_result.txt`
