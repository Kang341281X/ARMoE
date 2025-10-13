# Compression Artifacts Removal based on Frequency Domain Mixture of Experts

---

> **Abstract:** *In recent years, lossy compression standards such as H.264/AVC, H.265/HEVC, and H.266/VVC have been proposed and widely applied in image and video encoding. However, these compression algorithms inevitably introduce various complex types of compression artifacts, which severely degrade image quality. Although existing methods have attempted to remove artifacts through filter design or probabilistic prior modeling, they are often effective only for specific types of artifacts, lacking generalization and adaptability. To address this, we propose a novel image compression artifact removal model: ARMoE, which combines multiple frequency-domain transformations with the MoE (Mixture of Experts). Considering the frequency distribution and energy distribution differences of images, we introduce various frequency-domain transformations as expert branches and use a sparse activation mechanism to adaptively select the optimal frequency-domain expert to suppress compression artifacts, achieving an efficient artifact removal method. Furthermore, we re-encode and decode multiple high-quality datasets, including DF2K and Kodak24, using the VTM-20.0 codec under the VVC standard, constructing a more challenging artifact dataset. We conducted rigorous comparative experiments with current state-of-the-art image restoration methods, and the results demonstrate that ARMoE exhibits outstanding image restoration capability.* 



![](figs/ARMoE.png)

---



## Visual Results On Compression Artifacts Removal
![](figs/Vision.png)

---



## Environment and Installation

- Python 3.8
- PyTorch >= 1.8.0
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

```bash
git clone https://github.com/Kang341281X/ARMoE.git
conda create -n armoe python=3.8
conda activate armoe
pip install -r requirements.txt
python setup.py develop
```

---


## Datasets

Used training and testing sets can be downloaded as follows:

|                Task                |                         Training Set                         |                         Testing Set                          |
| :--------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|   Compression Artifacts Removal    | DIV2K (900 training images) + Flickr2K (2650 images) [complete training dataset DF2K: [Baidu Disk](https://pan.baidu.com/s/1nfmSVL98mImddhsuRilQFA?pwd=q2xe)] | Kodak24 + McMaster + CBSD68 [complete testing dataset: [Google Drive](https://drive.google.com/drive/folders/1rkUf276qRPjvf8HZZvJ2LseDNi4Ve-tD) / [Baidu Disk](https://pan.baidu.com/s/1-Ws_BUah-RwRlzOSPvFabQ?pwd=4gb3)] |
| Lightweight Image Super-Resolution | DIV2K (900 training images) [Google Drive](https://drive.google.com/file/d/1TubDkirxl4qAWelfOnpwaSKoj3KLAIG4/view?usp=share_link)] | Set5 + BSD100 [complete testing dataset: [Google Drive](https://drive.google.com/drive/folders/1EiCKNOluktInL7KXIMiGaHfsLru7repf) / [Baidu Disk](https://pan.baidu.com/s/1cnj8w5IMr7wEqrF8MkgS8w?pwd=4v5x)] |


### Run the following script to create VVC-CAR dataset:
The download address of the codec tool is https://drive.google.com/drive/folders/17VsQze69V1QnX0pdm60gm1c3-oqdi4wB

Place the VTM-20.0 codec and ffmpeg files in the same directory as the high-quality image folder.

```shell
# step1: Convert image from png format to yuv format
ffmpeg -i filename.png -s widthxheight -pix_fmt yuv420p filename.yuv

# step2: sing the All-Intra (AI) configuration in VTM-20.0
refer to: https://www.cnblogs.com/ayuanstudy/p/16085134.html

# step3: Execute encoding command
EncoderApp.exe   -c encoder_intra_vtm.cfg > Enc_Out.txt 

# step4: Execute decoding command
DecoderApp.exe   -b str.bin -o dec.yuv

# step5: Convert image from yuv format to png format
ffmpeg -f rawvideo -pix_fmt yuv420p -s:v widthxheight -pix_fmt yuv420p -i dec.yuv dec.png
```



---



## Models

| Method      | Params |                          Model Zoo                           |
| :---------- | :----: | :----------------------------------------------------------: |
| ARMoE       | 25.6M  | [Google Drive](https://drive.google.com/drive/folders/15CG48tVXHPcHiUwMXa6YLu8E1oieppnO) / [Baidu Disk](https://pan.baidu.com/s/1bfu_OtFuBRtuo66r8Ub7Uw?pwd=scer) |
| ARMoE-light |  845K  | [Google Drive](https://drive.google.com/drive/folders/1-AuGdGyV_O3TOvzNbUDlTsd9Nr678KsO) / [Baidu Disk](https://pan.baidu.com/s/1VmI-hFU3hE5WGgdIReeRcg?pwd=7yk6) |

---



## Training

- Run the following scripts. The training configuration is in `options/Train/`.

  ```shell
  # Compression Artifacts Removal
  python basicsr/train.py -opt options/Train/CAR/train_ARMoE_QP22.yml
  python basicsr/train.py -opt options/Train/CAR/train_ARMoE_QP27.yml
  python basicsr/train.py -opt options/Train/CAR/train_ARMoE_QP32.yml
  python basicsr/train.py -opt options/Train/CAR/train_ARMoE_QP37.yml
  
  # Lightweight Image Super-Resolution
  python basicsr/train.py -opt options/Train/SR/train_ARMoE_lightSRx2.yml
  ```

- The training experiment is in `experiments/`.



## Testing

- Download the pre-trained models and place them in `experiments/pretrained_models/`.

  We provide pre-trained models for Compression Artifacts Removal and Lightweight Image Super-Resolution: ARMoE (QP22, QP27, QP32, QP37) and ARMoE-light (x2).

- Download testing (Set5, BSD100) datasets, place them in `datasets/`.

- Run the following scripts. The testing configuration is in `options/Test/` .

  ```shell
  # Compression Artifacts Removal
  python basicsr/test.py -opt options/Test/CAR/test_ARMoE_QP22.yml
  python basicsr/test.py -opt options/Test/CAR/test_ARMoE_QP27.yml
  python basicsr/test.py -opt options/Test/CAR/test_ARMoE_QP32.yml
  python basicsr/test.py -opt options/Test/CAR/test_ARMoE_QP37.yml
  
  # Lightweight Image Super-Resolution
  python basicsr/test.py -opt options/Test/SR/test_ARMoE_lightSRx2.yml
  ```
  
- The output is in `results/`.

---



## Results

**Ablation Study**
<p align="center">
  <img width="800" src="figs/Ablation Study.png">
</p>


**ARMoE vs H.266/VVC**

<p align="center">
  <img width="600" src="figs/ARMoEvsVVC.png">
</p>


**Comparison on Different Loop Filter Configurations**
The test dataset is Kodak24, QP=22.
<p align="center">
  <img width="600" src="figs/LoopF.png">
</p>


**Comparison on VVC Compression Artifacts Removal**
<p align="center">
  <img width="900" src="figs/CAR.png">
</p>


**Floating Point Operations**
<p align="center">
  <img width="180" src="figs/FLOPs.png">
</p>


**Comparison on BD-Rate Reduction**
<p align="center">
  <img width="900" src="figs/BD-rate.png">
</p>


**Comparison on Lightweight Image Super-Resolution**
<p align="center">
  <img width="500" src="figs/LSR.png">
</p>





<!-- <details>
<summary>Ablation Study</summary>

<p align="center">
  <img width="500" src="figs/Ablation Study.png">
</p>
</details>


<details>
<summary>ARMoE vs H.266/VVC</summary>

<p align="center">
  <img width="500" src="figs/ARMoEvsVVC.png">
</p>
</details>


<details>
<summary>Comparison on Different Loop Filter Configurations</summary>
The test dataset is Kodak24, QP=22.
<p align="center">
  <img width="500" src="figs/LoopF.png">
</p>
</details>


<details>
<summary>Comparison on VVC Compression Artifacts Removal</summary>
<p align="center">
  <img width="500" src="figs/CAR.png">
</p>
</details>


<details>
<summary>Comparison on BD-Rate Reduction</summary>
<p align="center">
  <img width="500" src="figs/BD-rate.png">
</p>
</details>


<details>
<summary>Comparison on Lightweight Image Super-Resolution</summary>
<p align="center">
  <img width="500" src="figs/LSR.png">
</p>
</details> -->

---



## Acknowledgements

This code is built on  [BasicSR](https://github.com/XPixelGroup/BasicSR).
