# Note: If your work uses this algorithm or makes improvements based on it, please be sure to cite this paper. Thank you for your cooperation.

# 注意：如果您的工作用到了本算法，或者基于本算法进行了改进，请您务必引用本论文，谢谢配合。

# Point-MSD: Jointly Mamba Self-Supervised Self-Distilling Point Cloud Representation Learning

Jie Liu, Mengna Yang, Yu Tian, Da Song, Kang Li, Xin Cao※.

The 32th Pacific Conference on Computer Graphics and Applications (Pacific Graphics 2024), 2024. (Oral presentation)

## Installation

### 1. Dependencies

```bash
pip install -r requirements.txt

# Compile C++ extensions
cd ./extensions/chamfer_dist && python setup.py install --user

# install PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# install GPU KNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# install Mamba
pip install causal-conv1d>=1.1.0
pip install mamba-ssm
```

### 2. Datasets

Please download the used dataset with the following links:

- ShapeNet55 [https://drive.google.com/file/d/1jUB5yD7DP97-EqqU2A9mmr61JpNwZBVK/view?usp=sharing](https://drive.google.com/file/d/1jUB5yD7DP97-EqqU2A9mmr61JpNwZBVK/view?usp=sharing)
- ModelNet40 [https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip)
- ScanObjectNN [https://hkust-vgd.github.io/scanobjectnn](https://hkust-vgd.github.io/scanobjectnn)
- indoor3d_sem_seg_hdf5_data [https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip](https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip)
- Stanford3dDataset_v1.2_Aligned_Version [https://goo.gl/forms/4SoGp4KtH1jfRqEj2](https://goo.gl/forms/4SoGp4KtH1jfRqEj2)

### 3. Pre-training

```bash
./scripts/pretraining_shapenet.bash --data.in_memory true
```

### 4. Downstream

#### Classification on ModelNet40

```bash
./scripts/classification_modelnet40.bash --config configs/classification/_pretrained.yaml --model.pretrained_ckpt_path artifacts/point2vec-Pretraining-ShapeNet/XXXXXXXX/checkpoints/XXX.ckpt 
```

#### Classification on ScanObjectNN

```bash
./scripts/classification_scanobjectnn.bash --config configs/classification/_pretrained.yaml --model.pretrained_ckpt_path artifacts/point2vec-Pretraining-ShapeNet/XXXXXXXX/checkpoints/XXX.ckpt
```

#### Voting on ModelNet40

```bash
./scripts/voting_modelnet40.bash --finetuned_ckpt_path artifacts/point2vec-Pretraining-ShapeNet/XXXXXXXX/checkpoints/epoch=XXX-step=XXXXX-val_acc=0.XXX.ckpt
```

#### Semantic segmentation on S3DIS

```bash
./scripts/sem_part.bash --model.pretrained_ckpt_path artifacts/point2vec-Pretraining-ShapeNet/XXXXXXXX/checkpoints/XXX.ckpt
```

## Citation
If you find Point-MSD useful in your research, please consider citing:
```
@article{PointMSD,
  title={Point-MSD : Jointly Mamba Self-Supervised Self-Distilling Point Cloud Representation Learning},
  author={Linzhi Su, Mengna Yang, Jie Liu, Xingxing Hao, Kang Li, Xin Cao},
  year={2024},
}
```

## Acknowledgements 
We would like to thank and acknowledge referenced codes from the following repositories:

https://github.com/PangYatian/Point-MAE <br>
https://github.com/yichen928/STRL <br>
https://github.com/LMD0311/PointMamba <br>
https://point2vec.ka.codes/
