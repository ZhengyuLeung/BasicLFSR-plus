# 🧩 BasicLFSR-plus: LFSSR track
This track contains the implementation of Light Field (LF) Spatial Super-Resolution (SR) based on our BasicLFSR-plus framework.


## 📂 Dataset Preparation

We used the EPFL, HCInew, HCIold, INRIA and STFgantry datasets for both training and test. 
Please first download our datasets via [Baidu Drive](https://pan.baidu.com/s/1mYQR6OBXoEKrOk0TjV85Yw) (key:7nzy) or [OneDrive](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/zyliang_stu_xidian_edu_cn/EpkUehGwOlFIuSSdadq9S4MBEeFkNGPD_DlzkBBmZaV_mA?e=FiUeiv), and place the 5 datasets to the folder `./datasets/`.

Organize data as:
  ```
  datasets/
  ├── EPFL
  │    ├── training
  │    │    ├── Bench_in_Paris.mat
  │    │    ├── Billboards.mat
  │    │    └── ...
  │    ├── test
  │    │    ├── Bikes.mat
  │    │    ├── Books__Decoded.mat
  │    │    └── ...
  ├── HCI_new
  └── ...
  ```

Run `Generate_Data_for_SSR_Training.py` to generate training data. The generated data will be saved in `./data_for_training/` (SR_5x5_2x, SR_5x5_4x).
Run `Generate_Data_for_SSR_Test.py` to generate test data. The generated data will be saved in `./data_for_test/` (SR_5x5_2x, SR_5x5_4x).


## 🚀 Training

Modify the configs in `train_SSR.py` or use default arguments.

```
$ python train_SSR.py --model_name [model_name] --angRes 5 --scale_factor 2 --batch_size 8
$ python train_SSR.py --model_name [model_name] --angRes 5 --scale_factor 4 --batch_size 4
```

Checkpoints and Logs will be saved to `./log/`, and the `./log/` has the following structure:

```
log/
├── SR_5x5_2x
│    └── [dataset_name]
│         ├── [model_name]
│         │    ├── [model_name]_log.txt
│         │    ├── checkpoints
│         │    │    ├── [model_name]_5x5_2x_epoch_01_model.pth
│         │    │    ├── [model_name]_5x5_2x_epoch_02_model.pth
│         │    │    └── ...
│         │    ├── results
│         │    │    ├── VAL_epoch_01
│         │    │    ├── VAL_epoch_02
│         │    │    └── ...
│         ├── [other_model_name]
│         └── ...
├── SR_5x5_4x
└── ...
```


## 🧪 Test

Run `test_SSR.py` to perform network inference. Example for test [model_name] on 5x5 angular resolution for 2x/4xSR:
```
$ python test_SSR.py --model_name [model_name] --angRes 5 --scale_factor 2  
$ python test_SSR.py --model_name [model_name] --angRes 5 --scale_factor 4 
```

The PSNR and SSIM values of each dataset will be saved to **`./log/`**, and the **`./log/`** has the following structure:
```
log/
├── SR_5x5_2x
│    └── [dataset_name]
│        ├── [model_name]
│        │    ├── [model_name]_log.txt
│        │    ├── checkpoints
│        │    │   └── ...
│        │    └── results
│        │         ├── Test
│        │         │    ├── evaluation.xlsx
│        │         │    ├── [dataset_1_name]
│        │         │    │    ├── [scene_1_name]
│        │         │    │    │    ├── [scene_1_name]_CenterView.bmp
│        │         │    │    │    ├── [scene_1_name]_SAI.bmp
│        │         │    │    │    ├── views
│        │         │    │    │    │    ├── [scene_1_name]_0_0.bmp
│        │         │    │    │    │    ├── [scene_1_name]_0_1.bmp
│        │         │    │    │    │    ├── ...
│        │         │    │    │    │    └── [scene_1_name]_4_4.bmp
│        │         │    │    ├── [scene_2_name]
│        │         │    │    └── ...
│        │         │    ├── [dataset_2_name]
│        │         │    └── ...
│        │         ├── VAL_epoch_01
│        │         └── ...
│        ├── [other_model_name]
│        └── ...
├── SR_5x5_4x
└── ...
```


## 📊 Benchmark

We benchmark several methods on the above datasets. PSNR and SSIM metrics are used for quantitative evaluation.
To obtain the metric score for a dataset with `M` scenes, we first calculate the metric on `AxA` SAIs on each scene separately, then obtain the score for each scene by averaging its `A^2` scores, and finally obtain the score for this dataset by averaging the scores of all its `M` scenes.


<details>
<summary>📊 Click to expand benchmark results on 5x5 LF images for 2xSR </summary>

|    Methods    |  #Params. | EPFL | HCInew | HCIold | INRIA | STFgantry |
| :----------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | 
| **Bicubic**      |     -- | 29.740/0.9376 | 31.887/0.9356 | 37.686/0.9785 | 31.331/0.9577 | 31.063/0.9498 | 
| **VDSR**         | 0.665M | 32.498/0.9598 | 34.371/0.9561 | 40.606/0.9867 | 34.439/0.9741 | 35.541/0.9789 | 
| **EDSR**         | 38.62M | 33.089/0.9629 | 34.828/0.9592 | 41.014/0.9874 | 34.985/0.9764 | 36.296/0.9818 | 
| [**RCAN**](https://github.com/yulunzhang/RCAN)                 | 15.31M | 33.159/0.9634          | 35.022/0.9603         | 41.125/0.9875          | 35.046/0.9769         | 36.670/0.9831       | 
| [**resLF**](https://github.com/shuozh/resLF)                   | 7.982M | 33.617/0.9706          | 36.685/0.9739         | 43.422/0.9932          | 35.395/0.9804         | 38.354/0.9904       | 
| [**LFSSR**](https://github.com/jingjin25/LFSSR-SAS-PyTorch)    | 0.888M | 33.671/0.9744          | 36.802/0.9749         | 43.811/0.9938          | 35.279/0.9832         | 37.944/0.9898       | 
| [**LF-ATO**](https://github.com/jingjin25/LFSSR-ATO)           | 1.216M | 34.272/0.9757          | 37.244/0.9767         | 44.205/0.9942          | 36.170/0.9842         | 39.636/0.9929       | 
| [**LF_InterNet**](https://github.com/YingqianWang/LF-InterNet) | 5.040M | 34.112/0.9760          | 37.170/0.9763         | 44.573/0.9946          | 35.829/0.9843         | 38.435/0.9909       | 
| [**LF-DFnet**](https://github.com/YingqianWang/LF-DFnet)       | 3.940M | 34.513/0.9755          | 37.418/0.9773         | 44.198/0.9941          | 36.416/0.9840         | 39.427/0.9926       | 
| [**MEG-Net**](https://github.com/shuozh/MEG-Net)               | 1.693M | 34.312/0.9773          | 37.424/0.9777         | 44.097/0.9942          | 36.103/0.9849         | 38.767/0.9915       | 
| [**LF-IINet**](https://github.com/GaoshengLiu/LF-IINet)        | 4.837M | 34.732/0.9773          | 37.768/0.9790         | *44.852*/*0.9948*      | 36.566/0.9853         | 39.894/0.9936       | 
| [**DPT**](https://github.com/BITszwang/DPT)                    | 3.731M | 34.490/0.9758          | 37.355/0.9771         | 44.302/0.9943          | 36.409/0.9843         | 39.429/0.9926       | 
| [**LFT**](https://github.com/ZhengyuLiang24/LFT)               | 1.114M | *34.804*/*0.9781*      | *37.838*/*0.9791*     | 44.522/0.9945          | **36.594**/*0.9855*   | **40.510**/*0.9941* | 
| [**DistgSSR**](https://github.com/YingqianWang/DistgSSR)       | 3.532M | **34.809**/**0.9787**  | **37.959**/**0.9796** | **44.943**/**0.9949**  | *36.586*/**0.9859**   | *40.404*/**0.9942** |
</details>


<details>
<summary>📊 Click to expand benchmark results on 5x5 LF images for 4xSR </summary>

|    Methods    |  #Params. | EPFL | HCInew | HCIold | INRIA | STFgantry | 
| :----------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | 
| **Bicubic**      |     -- | 25.264/0.8324 | 27.715/0.8517 | 32.576/0.9344 | 26.952/0.8867 | 26.087/0.8452 | 
| **VDSR**         | 0.665M | 27.246/0.8777 | 29.308/0.8823 | 34.810/0.9515 | 29.186/0.9204|  28.506/0.9009 | 
| **EDSR**         | 38.89M | 27.833/0.8854 | 29.591/0.8869 | 35.176/0.9536 | 29.656/0.9257 | 28.703/0.9072 | 
| [**RCAN**](https://github.com/yulunzhang/RCAN)                  | 15.36M | 27.907/0.8863 | 29.694/0.8886 | 35.359/0.9548 | 29.805/0.9276 | 29.021/0.9131 | 
| [**resLF**](https://github.com/shuozh/resLF)                    | 8.646M | 28.260/0.9035 | 30.723/0.9107 | 36.705/0.9682 | 30.338/0.9412 | 30.191/0.9372 |
| [**LFSSR**](https://github.com/jingjin25/LFSSR-SAS-PyTorch)     | 1.774M | 28.596/0.9118 | 30.928/0.9145 | 36.907/0.9696 | 30.585/0.9467 | 30.570/0.9426 | 
| [**LF-ATO**](https://github.com/jingjin25/LFSSR-ATO)            | 1.364M | 28.514/0.9115 | 30.880/0.9135 | 36.999/0.9699 | 30.711/0.9484 | 30.607/0.9430 | 
| [**LF_InterNet**](https://github.com/YingqianWang/LF-InterNet)  | 5.483M | 28.812/0.9162 | 30.961/0.9161 | 37.150/0.9716 | 30.777/0.9491 | 30.365/0.9409 | 
| [**LF-DFnet**](https://github.com/YingqianWang/LF-DFnet)        | 3.990M | 28.774/0.9165 | 31.234/0.9196 | 37.321/0.9718 | 30.826/0.9503 | 31.147/0.9494 | 
| [**MEG-Net**](https://github.com/shuozh/MEG-Net)                | 1.775M | 28.749/0.9160 | 31.103/0.9177 | 37.287/0.9716 | 30.674/0.9490 | 30.771/0.9453 | 
| [**LF-IINet**](https://github.com/GaoshengLiu/LF-IINet)         | 4.886M | *29.038*/0.9188    | 31.331/0.9208         | *37.620*/*0.9734*     | *31.034*/0.9515       | 31.261/0.9502         | 
| [**DPT**](https://github.com/BITszwang/DPT)                     | 3.778M | 28.939/0.9170      | 31.196/0.9188         | 37.412/0.9721         | 30.964/0.9503         | 31.150/0.9488         |
| [**LFT**](https://github.com/ZhengyuLiang24/LFT)                | 1.163M | **29.255/0.9210**  | **31.462**/**0.9218** | **37.630**/**0.9735** | **31.205**/**0.9524** | **31.860**/**0.9548** | 
| [**DistgSSR**](https://github.com/YingqianWang/DistgSSR)        | 3.582M | 28.992/*0.9195*    | *31.380*/*0.9217*     | 37.563/0.9732         | 30.994/*0.9519*       | *31.649*/*0.9535*     | 
</details>


## ⬇️ Recources
* The pre-trained models of the aforementioned methods can be downlaoded via [this link](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/zyliang_stu_xidian_edu_cn/EtUBJ4eHG7BCjnUmtXpu9o0BvGVk5_v-RG95R_aRN46UwQ).



## 📬 Contact
For questions or contributions, feel free to reach out via email.
