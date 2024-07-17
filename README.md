# Vim

首先clone此项目

然后创建环境
```

conda create -n mamba python=3.10.13

conda activate mamba
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r vim/vim_requirements.txt
```

替换Mamba文件
```
#下载替代文件
wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.1.3.post1/causal_conv1d-1.1.3.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
wget https://github.com/state-spaces/mamba/releases/download/v1.1.1/mamba_ssm-1.1.1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install causal_conv1d-1.1.3.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl 
pip install mamba_ssm-1.1.1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl 


# 用conda env list 查看刚才安装的mamba环境的路径，我的mamba环境在/home/zju/anaconda3/envs/mamba
conda env list  
#用项目里的mamba_ssm替换安装在conda环境里的mamba_ssm
cp -rf mamba-1p1p1/mamba_ssm /home/zju/anaconda3/envs/mamba/lib/python3.10/site-packages
```

运行指令训练
```
#使用你的imagenet目录替换<path_to_IN1K_dataset>
#多个GPU训练
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --batch-size 128 --drop-path 0.0 --weight-decay 0.1 --num_workers 25 --data-path <path_to_IN1K_dataset> --output_dir ./output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --no_amp
#单GPU训练
python main.py --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --batch-size 128 --drop-path 0.0 --weight-decay 0.1 --num_workers 25 --data-path <path_to_IN1K_dataset> --output_dir ./output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --no_amp
```

