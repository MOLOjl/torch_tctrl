# Megaton-LM use

## install


```bash
# 首先完成前面的pytorch安装，建议新开个环境，和之前有冲突
# pytorch编译是共享的，因此不需要重新编译
# 官方安装是直接用docker的，但是所里的网下不下来那个docker，只能自己编译了

# apex安装
git clone https://github.com/NVIDIA/apex apex
cd apex
pip install packaging
# change libstdc++.so.6 if necessary [link to the right version of libstdc++.so with GLIBCXX_3.4.30]
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
# pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..

pip install --user -U nltk

# transformer engine
git clone --branch stable --recursive https://github.com/NVIDIA/TransformerEngine.git
git submodule update --init --recursive
pip install cmake
pip install .
cd ..


git clone https://github.com/Pegessi/Megatron-LM
git switch dtb
git submodule sync
git submodule update --init --recursive
pip install . # 只安装megatron.core
```

## train

直接执行单机多卡训练

注意所有的环境变量和配置都在里面设置

绝对路径修改一下，模型和数据集直接用我原来目录下的即可

```bash
./examples/pretrain_gpt_distributed.sh
```

## TODO

参考：https://zhuanlan.zhihu.com/p/668057319

跑通llama的训练（单节点2卡4卡训练就行）