# 🌱 BasicLFSR-plus


### <img src="https://raw.github.com/ZhengyuLiang24/BasicLFSR/main/figs/Thumbnail.jpg" width="1000">

Official PyTorch implementation of the IEEE TPAMI 2026 paper: "Diving into Epipolar Transformers for Light Field Super-Resolution and Disparity Estimation".

This repository provides an enhanced and unified toolbox for Light Field (LF) Image Super-Resolution (SR), supporting both:

- **LFSSR** – LF Spatial SR (improving resolution of each sub-view image)
- **LFASR** – LF Angular SR (increasing angular resolution and enabling novel view synthesis)

This is an extension of our previous work, **[BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR)**, which focused solely on spatial SR. 
It has proven to be a helpful toolbox for researchers to quickly get started with LF spatial SR and to facilitate the development of new algorithms. 

Looking ahead, **[BasicLFSR-plus](https://github.com/ZhengyuLeung/BasicLFSR-plus)**, together with our **[BasicLFDisp](https://github.com/ZhengyuLeung/BasicLFDisp)** repository for LF disparity estimation, aims to provide a more comprehensive and user-friendly benchmark toolbox for the LF research community. 

## ✨ News & Updates

- **[2026-03]** 🎉 Our paper **"Diving into Epipolar Transformers for Light Field Super-Resolution and Disparity Estimation"** has been accepted by *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*!
- **[2025-08]** 🚀 We have released the **BasicLFSR-plus** and **BasicLFDisp** toolbox, and provided the pre-trained models of our **EPIT** mechanism (i.e., EPIT-SSR, EPIT-ASR, and EPIT-Disp).


## 🔧 Installation
```bash
git clone https://github.com/ZhengyuLeung/BasicLFSR-plus.git
cd BasicLFSR-plus
pip install -r requirements.txt
```
💡 Tip: Make sure to use a Python virtual environment (e.g., conda or venv) to avoid package conflicts.


## 🚀 Getting Started

To get started with a specific task, please  refer to the corresponding README:

| Task  | Description | Instructions |
|---------|-------------|-------------|
| `lfssr` | Code, models and pretrained weights for **LF spatial SR** | See [`README_LFSSR`](https://github.com/ZhengyuLeung/BasicLFSR-plus/blob/main/README_LFSSR.md) |
| `lfasr` | Code, models and pretrained weights for **LF angular SR** | See [`README_LFASR`](https://github.com/ZhengyuLeung/BasicLFSR-plus/blob/main/README_LFASR.md) |




## 🤝 Contributions
Feel free to open pull requests or discussions!

Welcome to raise issues or email to [zyliang@nudt.edu.cn](zyliang@nudt.edu.cn) for any question regarding our BasicLFSR-plus.


## 🔗 Related Projects
- [BasicLFDisp](https://github.com/ZhengyuLeung/BasicLFDisp) 📖


## 📝 Citation
If you find this code or our paper useful for your research, please consider citing:
```
@article{EPIT2026,
 title = {Diving into Epipolar Transformers for Light Field Super-Resolution and Disparity Estimation},
 author = {Liang, Zhengyu and Wang, Yingqian and Wang, Longguang and Yang, Jungang and Guo, Yulan and Liu, Li and Zhou, Shilin and An, Wei},
 journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
 year = {2026},
}
```
