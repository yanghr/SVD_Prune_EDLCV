# Start up code

[TOC]



## train from scratch

### resnet-20

* channel

  ```
  CUDA_VISIBLE_DEVICES=2 nohup python3 cifar10_resnet20.py --save_path ./pretrained_model --decay 0 --perp_weight 1.0 --sensitivity 1e-12 --reg_type None --train --n_svd_s1 1>pretrain.out 2>&1 &
  ```

* spatial

  ```
  CUDA_VISIBLE_DEVICES=2 nohup python3 cifar10_resnet20.py --save_path ./pretrained_model_sp --decay 0 --perp_weight 1.0 --sensitivity 1e-12 --reg_type None --train --n_svd_s1 --dectype space 1>pretrain_sp.out 2>&1 &
  ```



### resnet-32

* channel

  ```
  CUDA_VISIBLE_DEVICES=2 nohup python3 cifar10_resnet20.py --save_path ./pretrained_model32 --decay 0 --perp_weight 1.0 --sensitivity 1e-12 --reg_type None --train --n_svd_s1 --depth 110 1>pretrain32.out 2>&1 &
  ```

* spatial

  ```
  CUDA_VISIBLE_DEVICES=2 nohup python3 cifar10_resnet20.py --save_path ./pretrained_model32_sp --decay 0 --perp_weight 1.0 --sensitivity 1e-12 --reg_type None --train --n_svd_s1 --dectype space --depth 32 1>pretrain32_sp.out 2>&1 &
  ```



### resnet-56

* channel

  ```
  CUDA_VISIBLE_DEVICES=2 nohup python3 cifar10_resnet20.py --save_path ./pretrained_model56 --decay 0 --perp_weight 1.0 --sensitivity 1e-12 --reg_type None --train --n_svd_s1 --depth 56 1>pretrain56.out 2>&1 &
  ```

* spatial

  ```
  CUDA_VISIBLE_DEVICES=2 nohup python3 cifar10_resnet20.py --save_path ./pretrained_model56_sp --decay 0 --perp_weight 1.0 --sensitivity 1e-12 --reg_type None --train --n_svd_s1 --dectype space --depth 56 1>pretrain56_sp.out 2>&1 &
  ```



### resnet-110

* channel

  ```
  CUDA_VISIBLE_DEVICES=2 nohup python3 cifar10_resnet20.py --save_path ./pretrained_model110 --decay 0 --perp_weight 1.0 --sensitivity 1e-12 --reg_type None --train --n_svd_s1 --depth 110 1>pretrain110.out 2>&1 &
  ```

* spatial

  ```
  CUDA_VISIBLE_DEVICES=2 nohup python3 cifar10_resnet20.py --save_path ./pretrained_model110_sp --decay 0 --perp_weight 1.0 --sensitivity 1e-12 --reg_type None --train --n_svd_s1 --dectype space --depth 110 1>pretrain110_sp.out 2>&1 &
  ```



### resnet-50(ImageNet)

* channel

  ```
  CUDA_VISIBLE_DEVICES=5,6 nohup python3 imagenet_resnet.py --save_path ./pretrained_model50 --decay 0 --perp_weight 1.0 --sensitivity 1e-12 --reg_type None --train --n_svd_s1 1>pretrain50.out 2>&1 &
  ```

  

* spatial

## train with pretraining

### resnet-20

#### channel

* Hoyer

  ```
  CUDA_VISIBLE_DEVICES=1 nohup python3 cifar10_resnet20.py --save_path ./Hoyer_Model/007 --load_path ./pretrained_model/SVD_Model.pth --decay 0.07 --perp_weight 1.0 -e 1e-4 --reg_type Hoyer --train --n_svd_s1 1>hoyer_007.out 2>&1 &
  ```

* Hoyer-Square

  ```
  CUDA_VISIBLE_DEVICES=3 nohup python3 cifar10_resnet20.py --save_path ./Hoyer-Square_Model --load_path ./pretrained_model/SVD_Model.pth --decay 1e-2 --perp_weight 1.0 --sensitivity 1e-1 --reg_type Hoyer-Square --train --n_svd_s1 1>hoyer_square.out 2>&1 &
  ```

* L1

  ```
  CUDA_VISIBLE_DEVICES=0 nohup python3 cifar10_resnet20.py --save_path ./L1_Model/03 --load_path ./pretrained_model/SVD_Model.pth --decay 3e-1 --perp_weight 1.0 -e 1e-4 --reg_type L1 --train --n_svd_s1 1>L1_03.out 2>&1 &
  ```



#### spatial

- Hoyer

  ```
  CUDA_VISIBLE_DEVICES=0 nohup python3 cifar10_resnet20.py --save_path ./Hoyer_Model_sp/01 --load_path ./pretrained_model_sp/SVD_Model.pth --decay 0.1 --perp_weight 1.0 -e 4e-6 --reg_type Hoyer --train --n_svd_s1 --dectype space 1>hoyer_sp_01.out 2>&1 &
  ```

- Hoyer-Square

  ```
  CUDA_VISIBLE_DEVICES=3 nohup python3 cifar10_resnet20.py --save_path ./Hoyer-Square_Model_sp --load_path ./pretrained_model_sp/SVD_Model.pth --decay 1e-2 --perp_weight 1.0 --sensitivity 3e-2 --reg_type Hoyer-Square --train --n_svd_s1 --dectype space 1>hoyer_square_sp.out 2>&1 &
  ```

- L1

  ```
  CUDA_VISIBLE_DEVICES=3 nohup python3 cifar10_resnet20.py --save_path ./L1_Model_sp/03 --load_path ./pretrained_model_sp/SVD_Model.pth --decay 0.3 --perp_weight 1.0 -e 1e-4 --reg_type L1 --train --n_svd_s1 --dectype space 1>L1_sp_03.out 2>&1 &
  ```



### resnet-32

#### channel

- Hoyer

  ```
  CUDA_VISIBLE_DEVICES=4 nohup python3 cifar10_resnet20.py --save_path ./Hoyer_Model32 --load_path ./pretrained_model32/SVD_Model.pth --decay 0.03 --perp_weight 1.0 --sensitivity 1e-2 --reg_type Hoyer --train --n_svd_s1 --depth 32 1>hoyer32.out 2>&1 &
  ```

- Hoyer-Square

  ```
  CUDA_VISIBLE_DEVICES=3 nohup python3 cifar10_resnet20.py --save_path ./Hoyer-Square_Model32 --load_path ./pretrained_model32/SVD_Model.pth --decay 0.007 --perp_weight 1.0 --sensitivity 0.03 --reg_type Hoyer-Square --train --n_svd_s1 --depth 32 1>hoyer_square32.out 2>&1 &
  ```

- L1

  ```
  CUDA_VISIBLE_DEVICES=4 nohup python3 cifar10_resnet20.py --save_path ./L1_Model32/01 --load_path ./pretrained_model32/SVD_Model.pth --decay 0.1 --perp_weight 1.0 -e 1e-4 --reg_type L1 --train --n_svd_s1 --depth 32 1>L1_32_01.out 2>&1 &
  ```



#### spatial

- Hoyer

  ```
  CUDA_VISIBLE_DEVICES=4 nohup python3 cifar10_resnet20.py --save_path ./Hoyer_Model32_sp/0005 --load_path ./pretrained_model32_sp/SVD_Model.pth --decay 0.005 --perp_weight 1.0 -e 5e-3 --reg_type Hoyer --train --n_svd_s1 --depth 32 --dectype space 1>hoyer32_sp_0005.out 2>&1 &
  ```

- Hoyer-Square

  ```
  CUDA_VISIBLE_DEVICES=2 nohup python3 cifar10_resnet20.py --save_path ./Hoyer-Square_Model32_sp/00003 --load_path ./pretrained_model32_sp/SVD_Model.pth --decay 3e-4 --perp_weight 1.0 -e 5e-4 --reg_type Hoyer-Square --train --n_svd_s1 --depth 32 --dectype space 1>hoyer_square32_sp_00003.out 2>&1 &
  ```

- L1

  ```
  CUDA_VISIBLE_DEVICES=0 nohup python3 cifar10_resnet20.py --save_path ./L1_Model32_sp/01 --load_path ./pretrained_model32_sp/SVD_Model.pth --decay 0.1 --perp_weight 1.0 -e 1e-4 --reg_type L1 --train --n_svd_s1 --depth 32 --dectype space 1>L1_32_sp_01.out 2>&1 &
  ```



### resnet-56

####channel

- Hoyer

  ```
  CUDA_VISIBLE_DEVICES=3 nohup python3 cifar10_resnet20.py --save_path ./Hoyer_Model56/0001 --load_path ./pretrained_model56/SVD_Model.pth --decay 0.001 --perp_weight 1.0 -e 5e-4 --reg_type Hoyer --train --n_svd_s1 --depth 56 1>hoyer56_0001.out 2>&1 &
  ```

- Hoyer-Square

  ```
  CUDA_VISIBLE_DEVICES=1 nohup python3 cifar10_resnet20.py --save_path ./Hoyer-Square_Model56/00001 --load_path ./pretrained_model56/SVD_Model.pth --decay 0.0001 --perp_weight 1.0 -e 1e-4 --reg_type Hoyer-Square --train --n_svd_s1 --depth 56 1>hoyer_square56_00001.out 2>&1 &
  ```

- L1

  ```
  CUDA_VISIBLE_DEVICES=1 nohup python3 cifar10_resnet20.py --save_path ./L1_Model56/03 --load_path ./pretrained_model56/SVD_Model.pth --decay 0.3 --perp_weight 1.0 -e 1e-1 --reg_type L1 --train --n_svd_s1 --depth 56 1>L1_56_03.out 2>&1 &
  ```



#### spatial

- Hoyer

  ```
  CUDA_VISIBLE_DEVICES=5 nohup python3 cifar10_resnet20.py --save_path ./Hoyer_Model56_sp/0003 --load_path ./pretrained_model56_sp/SVD_Model.pth --decay 0.003 --perp_weight 1.0 -e 1e-3 --reg_type Hoyer --train --n_svd_s1 --depth 56 --dectype space 1>hoyer56_sp_0003.out 2>&1 &
  ```

- Hoyer-Square

  ```
  CUDA_VISIBLE_DEVICES=1 nohup python3 cifar10_resnet20.py --save_path ./Hoyer-Square_Model56_sp/00001 --load_path ./pretrained_model56_sp/SVD_Model.pth --decay 0.0001 --perp_weight 1.0 -e 1e-4 --reg_type Hoyer-Square --train --n_svd_s1 --depth 56 --dectype space 1>hoyer_square56_sp_00001.out 2>&1 &
  ```

- L1

  ```
  CUDA_VISIBLE_DEVICES=1 nohup python3 cifar10_resnet20.py --save_path ./L1_Model56_sp/01 --load_path ./pretrained_model56_sp/SVD_Model.pth --decay 0.1 --perp_weight 1.0 -e 1e-1 --reg_type L1 --train --n_svd_s1 --depth 56 --dectype space 1>L1_56_sp_01.out 2>&1 &
  ```



### resnet-110

#### channel

- Hoyer

  ```
  CUDA_VISIBLE_DEVICES=5 nohup python3 cifar10_resnet20.py --save_path ./Hoyer_Model110/0001 --load_path ./pretrained_model110/SVD_Model.pth --decay 0.001 --perp_weight 1.0 -e 5e-4 --reg_type Hoyer --train --n_svd_s1 --depth 110 1>hoyer110_0001.out 2>&1 &
  ```

- Hoyer-Square

  ```
  CUDA_VISIBLE_DEVICES=5 nohup python3 cifar10_resnet20.py --save_path ./Hoyer-Square_Model110/00003 --load_path ./pretrained_model110/SVD_Model.pth --decay 0.0003 --perp_weight 1.0 -e 5e-4 --reg_type Hoyer-Square --train --n_svd_s1 --depth 110 1>hoyer_square110_00003.out 2>&1 &
  ```

- L1

  ```
  CUDA_VISIBLE_DEVICES=2 nohup python3 cifar10_resnet20.py --save_path ./L1_Model110/01 --load_path ./pretrained_model110/SVD_Model.pth --decay 0.1 --perp_weight 1.0 -e 1e-2 --reg_type L1 --train --n_svd_s1 --depth 110 1>L1_110_01.out 2>&1 &
  ```



#### spatial

- Hoyer

  ```
  CUDA_VISIBLE_DEVICES=6 nohup python3 cifar10_resnet20.py --save_path ./Hoyer_Model110_sp/001 --load_path ./pretrained_model110_sp/SVD_Model.pth --decay 0.01 --perp_weight 1.0 -e 2e-5 --reg_type Hoyer --train --n_svd_s1 --depth 110 --dectype space 1>hoyer110_sp_001.out 2>&1 &
  ```

- Hoyer-Square

  ```
  CUDA_VISIBLE_DEVICES=6 nohup python3 cifar10_resnet20.py --save_path ./Hoyer-Square_Model110_sp/00003 --load_path ./pretrained_model110_sp/SVD_Model.pth --decay 0.0003 --perp_weight 1.0 -e 5e-4 --reg_type Hoyer-Square --train --n_svd_s1 --depth 110 --dectype space 1>hoyer_square110_sp_00003.out 2>&1 &
  ```

- L1

  ```
  CUDA_VISIBLE_DEVICES=0 nohup python3 cifar10_resnet20.py --save_path ./L1_Model110_sp/03 --load_path ./pretrained_model110_sp/SVD_Model.pth --decay 0.3 --perp_weight 1.0 -e 1e-2 --reg_type L1 --train --n_svd_s1 --depth 110 --dectype space 1>L1_110_sp_03.out 2>&1 &
  ```



## test and prune

### resnet-20

#### channel

* None

  ```
  python3 cifar10_resnet20.py --load_path ./pretrained_model/SVD_Model.pth --save_path ./pretrained_model -e 3e-1 --test --n_svd_s1
  ```

* Hoyer

  ```
  CUDA_VISIBLE_DEVICES=5 python3 cifar10_resnet20.py --load_path ./Hoyer_Model/01/SVD_Model.pth --save_path ./Hoyer_Model/01 -e 3e-6 --test --n_svd_s1
  ```

* Hoyer-Square

  ```
  CUDA_VISIBLE_DEVICES=2 python3 cifar10_resnet20.py --load_path ./Hoyer-Square_Model/0007/SVD_Model.pth --save_path ./Hoyer-Square_Model/0007 -e 5e-6 --test --n_svd_s1
  ```

* L1

  ```
  CUDA_VISIBLE_DEVICES=2 python3 cifar10_resnet20.py --load_path ./L1_Model/03/SVD_Model.pth --save_path ./L1_Model/03 -e 1e-1 --test --n_svd_s1
  ```



#### spatial

* None

  ```
  python3 cifar10_resnet20.py --load_path ./pretrained_model_sp/SVD_Model.pth --save_path ./pretrained_model_sp -e 3e-1 --test --n_svd_s1 --dectype space
  ```

* Hoyer

  ```
  python3 cifar10_resnet20.py --load_path ./Hoyer_Model_sp/001/SVD_Model.pth --save_path ./Hoyer_Model_sp/001 -e 1e-1 --test --n_svd_s1 --dectype space
  ```

* Hoyer-Square

  ```
  python3 cifar10_resnet20.py --load_path ./Hoyer-Square_Model_sp/SVD_Model.pth --save_path ./Hoyer-Square_Model_sp --sensitivity 0.1 --test --n_svd_s1 --dectype space
  ```

* L1

  ```
  python3 cifar10_resnet20.py --load_path ./L1_Model_sp/03/SVD_Model.pth --save_path ./L1_Model_sp/03 -e 1e-1 --test --n_svd_s1 --dectype space
  ```



### resnet-32

#### channel

* Hoyer

  ```
  python3 cifar10_resnet20.py --load_path ./Hoyer_Model32/0003/SVD_Model.pth --save_path ./Hoyer_Model32/0003 -e 5e-3 --test --n_svd_s1 --depth 32
  ```

* Hoyer-Square

  ```
  python3 cifar10_resnet20.py --load_path ./Hoyer-Square_Model32/0001/SVD_Model.pth --save_path ./Hoyer-Square_Model32/0001 -e 1e-3 --test --n_svd_s1 --depth 32
  ```

* L1

  ```
  python3 cifar10_resnet20.py --load_path ./L1_Model32/03/SVD_Model.pth --save_path ./L1_Model32/03 -e 5e-2 --test --n_svd_s1 --depth 32
  ```



#### spatial

* Hoyer

  ```
  python3 cifar10_resnet20.py --load_path ./Hoyer_Model32_sp/0001/SVD_Model.pth --save_path ./Hoyer_Model32_sp/0001 -e 5e-4 --test --n_svd_s1 --depth 32 --dectype space
  ```

* Hoyer-Square

  ```
  python3 cifar10_resnet20.py --load_path ./Hoyer-Square_Model32_sp/00003/SVD_Model.pth --save_path ./Hoyer-Square_Model32_sp/00003 -e 5e-3 --test --n_svd_s1 --depth 32 --dectype space
  ```

* L1

  ```
  python3 cifar10_resnet20.py --load_path ./L1_Model32_sp/01/SVD_Model.pth --save_path ./L1_Model32_sp/01 -e 1e-2 --test --n_svd_s1 --depth 32 --dectype space
  ```



### resnet-56

#### channel

* None

  ```
  python3 cifar10_resnet20.py --load_path ./pretrained_model56/SVD_Model.pth --save_path ./pretrained_model56 -e 3e-1 --test --n_svd_s1 --depth 56
  ```

* Hoyer

  ```
  python3 cifar10_resnet20.py --load_path ./Hoyer_Model56/0001/SVD_Model.pth --save_path ./Hoyer_Model56/0001 -e 5e-3 --test --n_svd_s1 --depth 56
  ```

* Hoyer-Square

  ```
  python3 cifar10_resnet20.py --load_path ./Hoyer-Square_Model56/00001/SVD_Model.pth --save_path ./Hoyer-Square_Model56/00001 -e 7e-3 --test --n_svd_s1 --depth 56
  ```

* L1

  ```
  CUDA_VISIBLE_DEVICES=1 python3 cifar10_resnet20.py --load_path ./L1_Model56/01/SVD_Model.pth --save_path ./L1_Model56/01 -e 1e-1 --test --n_svd_s1 --depth 56
  ```



#### spatial

* None

  ```
  python3 cifar10_resnet20.py --load_path ./pretrained_model56_sp/SVD_Model.pth --save_path ./pretrained_model56_sp -e 3e-1 --test --n_svd_s1 --depth 56 --dectype space
  ```

* Hoyer

  ```
  python3 cifar10_resnet20.py --load_path ./Hoyer_Model56_sp/0001/SVD_Model.pth --save_path ./Hoyer_Model56_sp/0001 -e 1e-1 --test --n_svd_s1 --depth 56 --dectype space
  ```

* Hoyer-Square

  ```
  python3 cifar10_resnet20.py --load_path ./Hoyer-Square_Model56_sp/00001/SVD_Model.pth --save_path ./Hoyer-Square_Model56_sp/00001 -e 1e-1 --test --n_svd_s1 --depth 56 --dectype space
  ```

* L1

  ```
  python3 cifar10_resnet20.py --load_path ./L1_Model56_sp/003/SVD_Model.pth --save_path ./L1_Model56_sp/003 -e 1e-1 --test --n_svd_s1 --depth 56 --dectype space
  ```



### resnet-110

#### channel

* Hoyer

  ```
  python3 cifar10_resnet20.py --load_path ./Hoyer_Model110/0001/SVD_Model.pth --save_path ./Hoyer_Model110/0001 -e 1e-2 --test --n_svd_s1 --depth 110
  ```

* Hoyer-Square

  ```
  python3 cifar10_resnet20.py --load_path ./Hoyer-Square_Model110/00003/SVD_Model.pth --save_path ./Hoyer-Square_Model110/00003 -e 1e-3 --test --n_svd_s1 --depth 110
  ```

* L1

  ```
  python3 cifar10_resnet20.py --load_path ./L1_Model110/003/SVD_Model.pth --save_path ./L1_Model110/003 -e 3e-1 --test --n_svd_s1 --depth 110
  ```



#### spatial

* Hoyer

  ```
  python3 cifar10_resnet20.py --load_path ./Hoyer_Model110_sp//0001/SVD_Model.pth --save_path ./Hoyer_Model110_sp/0001 -e 1e-3 --test --n_svd_s1 --depth 110 --dectype space
  ```

* Hoyer-Square

  ```
  python3 cifar10_resnet20.py --load_path ./Hoyer-Square_Model110_sp/00003/SVD_Model.pth --save_path ./Hoyer-Square_Model110_sp/00003 -e 1e-3 --test --n_svd_s1 --depth 110 --dectype space
  ```

* L1

  ```
  CUDA_VISIBLE_DEVICES=5 python3 cifar10_resnet20.py --load_path ./L1_Model110_sp/003/SVD_Model.pth --save_path ./L1_Model110_sp/003 -e 5e-2 --test --n_svd_s1 --depth 110 --dectype space
  ```



## fine tune

### resnet 20

#### channel

* Hoyer

  ```
  CUDA_VISIBLE_DEVICES=2 nohup python3 cifar10_resnet20.py --save_path ./Hoyer_FT_Model/01 --load_path ./Hoyer_Model/01/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun 1>hoyerFT_01.out 2>&1 &
  ```

* Hoyer-Square

  ```
  CUDA_VISIBLE_DEVICES=2 nohup python3 cifar10_resnet20.py --save_path ./Hoyer-Square_FT_Model --load_path ./Hoyer-Square_Model/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun 1>hoyerSFT.out 2>&1 &
  ```

* L1

  ```
  CUDA_VISIBLE_DEVICES=2 nohup python3 cifar10_resnet20.py --save_path ./L1_FT_Model/03 --load_path ./L1_Model/03/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun 1>L1FT_03.out 2>&1 &
  ```



#### spatial

* None

  ```
  CUDA_VISIBLE_DEVICES=0 nohup python3 cifar10_resnet20.py --save_path ./None_FT_Model_sp --load_path ./pretrained_model_sp/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun --dectype space 1>NoneFT_sp.out 2>&1 &
  ```

* Hoyer

  ```
  CUDA_VISIBLE_DEVICES=0 nohup python3 cifar10_resnet20.py --save_path ./Hoyer_FT_Model_sp/001 --load_path ./Hoyer_Model_sp/001/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun --dectype space 1>hoyerFT_sp.out_001 2>&1 &
  ```

* Hoyer-Square

  ```
  CUDA_VISIBLE_DEVICES=2 nohup python3 cifar10_resnet20.py --save_path ./Hoyer-Square_FT_Model_sp --load_path ./Hoyer-Square_Model_sp/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun --dectype space 1>hoyerSFT_sp.out 2>&1 &
  ```

* L1

  ```
  CUDA_VISIBLE_DEVICES=2 nohup python3 cifar10_resnet20.py --save_path ./L1_FT_Model_sp/03 --load_path ./L1_Model_sp/03/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun --dectype space 1>L1FT_sp_03.out 2>&1 &
  ```



### resnet 32

#### channel

* Hoyer

  ```
  CUDA_VISIBLE_DEVICES=3 nohup python3 cifar10_resnet20.py --save_path ./Hoyer_FT_Model32/001 --load_path ./Hoyer_Model32/001/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun --depth 32 1>hoyerFT32_001.out 2>&1 &
  ```

* Hoyer-Square

  ```
  CUDA_VISIBLE_DEVICES=3 nohup python3 cifar10_resnet20.py --save_path ./Hoyer-Square_FT_Model32/0001 --load_path ./Hoyer-Square_Model32/0001/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun --depth 32 1>hoyerSFT32_0001.out 2>&1 &
  ```

* L1

  ```
  CUDA_VISIBLE_DEVICES=3 nohup python3 cifar10_resnet20.py --save_path ./L1_FT_Model32/003 --load_path ./L1_Model32/003/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun --depth 32 1>L1FT32_003.out 2>&1 &
  ```



#### spatial

* Hoyer

  ```
  CUDA_VISIBLE_DEVICES=2 nohup python3 cifar10_resnet20.py --save_path ./Hoyer_FT_Model32_sp/0001 --load_path ./Hoyer_Model32_sp/0001/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun --depth 32 --dectype space 1>hoyerFT32_sp_0001.out 2>&1 &
  ```

* Hoyer-Square

  ```
  CUDA_VISIBLE_DEVICES=2 nohup python3 cifar10_resnet20.py --save_path ./Hoyer-Square_FT_Model32_sp/00003 --load_path ./Hoyer-Square_Model32_sp/00003/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun --depth 32 --dectype space 1>hoyerSFT32_sp_00003.out 2>&1 &
  ```

* L1

  ```
  CUDA_VISIBLE_DEVICES=5 nohup python3 cifar10_resnet20.py --save_path ./L1_FT_Model32_sp/01 --load_path ./L1_Model32_sp/01/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun --depth 32 --dectype space 1>L1FT32_sp_01.out 2>&1 &
  ```



### resnet 56

#### channel

* Hoyer

  ```
  CUDA_VISIBLE_DEVICES=3 nohup python3 cifar10_resnet20.py --save_path ./Hoyer_FT_Model56/0001 --load_path ./Hoyer_Model56/0001/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun --depth 56 1>hoyerFT56_0001.out 2>&1 &
  ```

* Hoyer-Square

  ```
  CUDA_VISIBLE_DEVICES=0 nohup python3 cifar10_resnet20.py --save_path ./Hoyer-Square_FT_Model56/00001 --load_path ./Hoyer-Square_Model56/00001/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun --depth 56 1>hoyerSFT56_00001.out 2>&1 &
  ```

* L1

  ```
  CUDA_VISIBLE_DEVICES=1 nohup python3 cifar10_resnet20.py --save_path ./L1_FT_Model56/01 --load_path ./L1_Model56/01/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun --depth 56 1>L1FT56_01.out 2>&1 &
  ```



#### spatial

* None

  ```
  CUDA_VISIBLE_DEVICES=2 nohup python3 cifar10_resnet20.py --save_path ./None_FT_Model56_sp --load_path ./pretrained_model56_sp/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun --depth 56 --dectype space 1>NoneFT56_sp.out 2>&1 &
  ```

* Hoyer

  ```
  CUDA_VISIBLE_DEVICES=3 nohup python3 cifar10_resnet20.py --save_path ./Hoyer_FT_Model56_sp/0003 --load_path ./Hoyer_Model56_sp/0003/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun --depth 56 --dectype space 1>hoyerFT56_sp_0003.out 2>&1 &
  ```

* Hoyer-Square

  ```
  CUDA_VISIBLE_DEVICES=0 nohup python3 cifar10_resnet20.py --save_path ./Hoyer-Square_FT_Model56_sp/00001 --load_path ./Hoyer-Square_Model56_sp/00001/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun --depth 56 --dectype space 1>hoyerSFT56_sp_00001.out 2>&1 &
  ```

* L1

  ```
  CUDA_VISIBLE_DEVICES=4 nohup python3 cifar10_resnet20.py --save_path ./L1_FT_Model56_sp/003 --load_path ./L1_Model56_sp/003/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun --depth 56 --dectype space 1>L1FT56_sp_003.out 2>&1 &
  ```



### resnet 110

#### channel

* Hoyer

  ```
  CUDA_VISIBLE_DEVICES=5 nohup python3 cifar10_resnet20.py --save_path ./Hoyer_FT_Model110/0001 --load_path ./Hoyer_Model110/0001/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun --depth 110 1>hoyerFT110_0001.out 2>&1 &
  ```

* Hoyer-Square

  ```
  CUDA_VISIBLE_DEVICES=5 nohup python3 cifar10_resnet20.py --save_path ./Hoyer-Square_FT_Model110/00003 --load_path ./Hoyer-Square_Model110/00003/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun --depth 110 1>hoyerSFT110_00003.out 2>&1 &
  ```

* L1

  ```
  CUDA_VISIBLE_DEVICES=6 nohup python3 cifar10_resnet20.py --save_path ./L1_FT_Model110/003 --load_path ./L1_Model110/003/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun --depth 110 1>L1FT110_003.out 2>&1 &
  ```



#### spatial

* Hoyer

  ```
  CUDA_VISIBLE_DEVICES=6 nohup python3 cifar10_resnet20.py --save_path ./Hoyer_FT_Model110_sp/0001 --load_path ./Hoyer_Model110_sp/0001/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun --depth 110 --dectype space 1>hoyerFT110_sp_0001.out 2>&1 &
  ```

* Hoyer-Square

  ```
  CUDA_VISIBLE_DEVICES=6 nohup python3 cifar10_resnet20.py --save_path ./Hoyer-Square_FT_Model110_sp/00003 --load_path ./Hoyer-Square_Model110_sp/00003/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun --depth 110 --dectype space 1>hoyerSFT110_sp_00003.out 2>&1 &
  ```

* L1

  ```
  CUDA_VISIBLE_DEVICES=5 nohup python3 cifar10_resnet20.py --save_path ./L1_FT_Model110_sp/003 --load_path ./L1_Model110_sp/003/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun --depth 110 --dectype space 1>L1FT110_sp_003.out 2>&1 &
  ```