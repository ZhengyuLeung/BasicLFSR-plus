# 🧩 BasicLFSR-plus: LFASR track
This track contains the implementation of Light Field (LF) ANgular Super-Resolution (SR) based on our BasicLFSR-plus framework.


## 📂 Dataset Preparation

We used the RE_HCI and RE_Lytro datasets for both training and test. 
Please first download our datasets via [OneDrive](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/zyliang_stu_xidian_edu_cn/EklAQ0a4ftJLvEfjZ64UoWgBd5he4N37_VSM9u41XfocDQ), and place the 5 datasets to the folder `./datasets/`.

Organize data as:
  ```
  datasets/
  ├── RE_HCI
  │    ├── train
  │    │    ├── antinuous.mat
  │    │    ├── boardgames.mat
  │    │    └── ...
  │    └── test
  │         ├──HCInew
  │         │    ├── bedroom.mat
  │         │    ├── bicycle.mat
  │         │    └── ...
  │         └──HCIold
  │              └── ...
  └── RE_Lytro
  ```

Run `Generate_Data_for_ASR_Training.py` to generate training data. The generated data will be saved in `./data_for_training/` (ASR).

Run `Generate_Data_for_ASR_Test.py` to generate test data. The generated data will be saved in `./data_for_test/` (ASR).


## 🚀 Training

Modify the configs in `train_ASR.py` or use default arguments.

```
$ python train_ASR.py --model_name [model_name] --angRes_in 2 --angRes_out 7
```

Checkpoints and Logs will be saved to `./log/`, and the `./log/` has the following structure:

```
log/ASR_2x2_7x7
├── [model_name]
│    ├── [model_name]_log.txt
│    ├── checkpoints
│    │    ├── [model_name]_2x2_7x7_ASR_epoch_01_model.pth
│    │    ├── [model_name]_2x2_7x7_ASR_epoch_02_model.pth
│    │    └── ...
│    └── results
│         ├── VAL_epoch_01
│         ├── VAL_epoch_02
│         └── ...
├── [other_model_name]
└── ...
```


## 🧪 Test

Run `test_ASR.py` to perform network inference. Example for test [model_name] on `RE_HCI_new` and `RE_HCI_old` for 2x2->7x7 SR:
```
$ python test_SSR.py --model_name [model_name] --angRes_in 2 --angRes_out 7  --data_list_for_test ['RE_HCI_new', 'RE_HCI_old'] 
```

The PSNR and SSIM values of each dataset will be saved to **`./log/`**, and the **`./log/`** has the following structure:
```
log/ASR_2x2_7x7
├── [model_name]
│    ├── [model_name]_log.txt
│    ├── checkpoints
│    │   └── ...
│    └── results
│         ├── Test
│         │    ├── evaluation.xlsx
│         │    ├── [dataset_1_name]
│         │    │    ├── [scene_1_name]
│         │    │    │    ├── [scene_1_name]_SAI.bmp
│         │    │    │    ├── [scene_1_name]_error.bmp
│         │    │    │    └── views
│         │    │    │         ├── [scene_1_name]_0_0.bmp
│         │    │    │         ├── [scene_1_name]_0_1.bmp
│         │    │    │         ├── ...
│         │    │    │         └── [scene_1_name]_4_4.bmp
│         │    │    ├── [scene_2_name]
│         │    │    └── ...
│         │    ├── [dataset_2_name]
│         │    └── ...
│         ├── VAL_epoch_01
│         └── ...
├── [other_model_name]
└── ...
```


## 📊 Benchmark

We benchmark several methods on the above datasets. PSNR and SSIM metrics are used for quantitative evaluation.
The metric score for each scene was by averaging the scores of reconstructed views (total 45 views for 2x2->7x7 ASR). 


<details>
<summary>📊 Click to expand benchmark results on 5x5 LF images for 2xSR </summary>


<table>
  <thead>
    <tr>
      <th rowspan="2" style="text-align:center">Methods</th>
      <th rowspan="2" style="text-align:center">#Params.</th>
      <th colspan="2" style="text-align:center">Synthetic Test Sets</th>
      <th colspan="3" style="text-align:center">Real-World Test Sets</th>
    </tr>
    <tr>
      <th>HCInew</th>
      <th>HCIold</th>
      <th>30scenes</th>
      <th>Occlusions</th>
      <th>Reflective</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Kalantari2016</td>
      <td>1.63M</td>
      <td>28.91/0.9150</td>
      <td>34.93/0.9405</td>
      <td>37.49/0.9839</td>
      <td>33.45/0.9689</td>
      <td>35.10/0.9709</td>
    </tr>
    <tr>
      <td>LFASR-geo</td>
      <td>1.01M</td>
      <td>32.23/0.9555</td>
      <td>38.40/0.9707</td>
      <td>38.45/0.9853</td>
      <td>35.28/0.9795</td>
      <td>35.46/0.9739</td>
    </tr>
    <tr>
      <td>FS-GAF</td>
      <td>1.54M</td>
      <td>34.28/0.9725</td>
      <td>39.21/0.9808</td>
      <td>40.52/0.9918</td>
      <td>36.77/0.9849</td>
      <td>36.84/0.9777</td>
    </tr>
    <tr>
      <td>Yeung2018</td>
      <td>0.88M</td>
      <td>32.09/0.9430</td>
      <td>40.44/0.9619</td>
      <td>42.21/0.9933</td>
      <td>38.09/0.9872</td>
      <td>38.35/0.9802 </td>
    </tr>
    <tr>
      <td>Pseudo4DCNN</td>
      <td>0.09M</td>
      <td>27.59/0.8752</td>
      <td>35.52/0.9327</td>
      <td>38.20/0.9758 </td>
      <td>35.55/0.9756</td>
      <td>37.18/0.9745</td>
    </tr>
    <tr>
      <td>P4DCNN</td>
      <td>0.26M</td>
      <td>29.56/0.9113</td>
      <td>36.60/0.9467</td>
      <td>39.75/0.9860</td>
      <td>36.52/0.9813</td>
      <td>37.56/0.9764</td>
    </tr>
    <tr>
      <td>SAA-Net</td>
      <td>0.74M</td>
      <td>30.10/0.9230</td>
      <td>36.14/0.9490</td>
      <td>39.77/0.9921</td>
      <td>36.68/0.9850</td>
      <td>37.42/0.9775</td>
    </tr>
    <tr>
      <td>DistgASR</td>
      <td>2.68M</td>
      <td>34.70/0.9735</td>
      <td>42.18/0.9782</td>
      <td>43.49/0.9952</td>
      <td>39.41/0.9905</td>
      <td>39.15/0.9797</td>
    </tr>
    <tr>
      <td>LFSAV</td>
      <td>1.39M</td>
      <td>32.44/0.9513</td>
      <td>41.14/0.9673</td>
      <td>42.72/0.9941</td>
      <td>38.52/0.9884</td>
      <td>38.75/0.9817</td>
    </tr>
    <tr>
      <td>LF-EASR</td>
      <td>6.77M</td>
      <td>33.65/0.9638</td>
      <td>40.83/0.9681</td>
      <td>41.47/0.9918</td>
      <td>37.65/0.9863</td>
      <td>38.21/0.9794</td>
    </tr>
    <tr>
      <td>EPIT-ASR</td>
      <td>1.03M</td>
      <td>34.76/0.9676</td>
      <td>42.26/0.9869</td>
      <td>43.54/0.9951</td>
      <td>39.43/0.9907</td>
      <td>39.19/0.9809</td>
    </tr>
  </tbody>
</table>




</details>



## ⬇️ Recources
* The pre-trained models of the aforementioned methods can be downlaoded via [this link]().



## 📬 Contact
For questions or contributions, feel free to reach out via email.
