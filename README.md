# Image De-raining papers

![](https://img.shields.io/badge/recent:update-2021 Oct.-red) ![](https://img.shields.io/badge/PaperNumber-37-brightgreen) ![](https://img.shields.io/badge/PRs-Welcome-red) ![](https://img.shields.io/badge/Issues-Welcome-red) 

Must-read papers on Image de-raining which include recent prior based and learning based methods. The paper list is mainly maintained by  [Schizophreni](https://github.com/Schizophreni/). We have merged the paper listed in [DerainZoo](https://github.com/nnUyi/DerainZoo) and re-organized recent papers for better comparison and understanding.  Note that this list is also friendly for writing introduction or related work of your academic paper. 

## Contents

- [Image de-raining papers](#derainpapers)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Papers](#papers)
    - [Survey](#Survey)
    - [Learning based](#Learning based)
      - [Linear Decompostion](#Linear Decomposition)
      - [Generation Model](#Generation Model)
      - [Recurrent Model](#Recurrent Model)
    - [Prior Based](#Prior Based)
    - [Hybrid](#hybrid)
    - [Image de-raining meets high level vision](#High Level)
  - [Other Contributors](#other-contributors)



## Introduction

This is a paper list about *image de-raining* researches. Image de-raining focuses on restoring the clean background given the rain-contaminated images as input. The basic assumption for image de-raining is that the information required for recovering the degraded pixels can be extracted from its neighbors.

## Papers

### Survey

1. **Single Image Deraining: From Model-Based to Data-Driven and Beyond.**  TPAMI. 

   *Yang Wenhan, T. Tan Robby, Wang Shiqi, Fang Yuming, and Liu Jiaying.*  [[pdf](https://arxiv.org/pdf/1912.07150.pdf)], [[cite]]([Single Image Deraining: From Model-Based to Data-Driven... - Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Single+Image+Deraining%3A+From+Model-Based+to+Data-Driven+and+Beyond&btnG=)), 2021. 

### Learning Based

- <h4>Linear Decomposition</h4>

  1. **Structure-Preserving Deraining with Residue Channel Prior Guidance. (SPDNet)** ICCV. ![](https://img.shields.io/badge/single image-purple) ![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre train-green)

     *Yi Qiaosi, Li Juncheng, Dai Qinyan, Fang Faming, Zhang Guixu, and Zeng Tieyong*. [[pdf]](https://junchenglee.com/paper/ICCV_2021.pdf), [[github]](https://github.com/Joyies/SPDNet), [[cite]]([Structure-Preserving Deraining with Residue Channel... - Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Structure-Preserving+Deraining+with+Residue+Channel+Prior+Guidance&btnG=)), 2021.

  2.  **Removing Raindrops and Rain Streaks in One Go. (CCN)** CVPR. ![](https://img.shields.io/badge/single image-purple)

     *Quan Ruijie, Yu Xin, Liang Yuanzhi, and Yang Yi*. [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Quan_Removing_Raindrops_and_Rain_Streaks_in_One_Go_CVPR_2021_paper.pdf), [[cite]]([**Removing Raindrops and Rain Streaks in One Go - Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=**Removing+Raindrops+and+Rain+Streaks+in+One+Go&btnG=)), 2021.

  3. **Rain Streak Removal via Dual Graph Convolutional Network. (DualGCN)** AAAI. ![](https://img.shields.io/badge/single image-purple) ![](https://img.shields.io/badge/project-blue)

     *Fu Xueyang, Qi Qi, Zha Zheng-Jun, Zhu Yurui, and Ding Xinghao*. [[pdf]](https://www.aaai.org/AAAI21Papers/AAAI-228.FuXY.pdf), [[github]](https://xueyangfu.github.io/paper/2021/AAAI/code.zip), [[cite]]([Rain Streak Removal via Dual Graph Convolutional Network. - Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Rain+Streak+Removal+via+Dual+Graph+Convolutional+Network.&btnG=)), 2021. 

  4. **Pre-Trained Image Processing Transformer. (IPT)** CVPR. ![](https://img.shields.io/badge/single image-purple) ![](https://img.shields.io/badge/transformer-black)

     *Chen Hanting, Wang Yunhe, Guo Tianyu, Xu Chang, Deng Yipeng, Liu Zhenhua, Ma Siwei, Xu Chunjing, Xu Chao, and Gao Wen*. [[pdf]]([2012.00364.pdf (arxiv.org)](https://arxiv.org/pdf/2012.00364.pdf)), [[cite]]([Pre-Trained Image Processing Transformer - Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Pre-Trained+Image+Processing+Transformer&btnG=)), 2021.

  5.  **Self-Learning Video Rain Streak Removal: When Cyclic Consistency Meets Temporal Correspondence. (SLDNet)** CVPR. ![](https://img.shields.io/badge/video-orange)

     *Wang Hong, Xie Qi, Zhao Qian, and Meng Deyu*. [[pdf]]([A Model-Driven Deep Neural Network for Single Image Rain Removal (thecvf.com)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_A_Model-Driven_Deep_Neural_Network_for_Single_Image_Rain_Removal_CVPR_2020_paper.pdf)), [[cite]]([Self-Learning Video Rain Streak Removal: When Cyclic... - Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Self-Learning+Video+Rain+Streak+Removal%3A+When+Cyclic+Consistency+Meets+Temporal+Correspondence.&btnG=)), 2020.
     
  6. 

     ****

- <h4> Generation Model</h4>

  1. **Semi-Supervised Video Deraining with Dynamical Rain Generator. (S2VD) ** CVPR. ![](https://img.shields.io/badge/video-orange) ![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre train-green) 

     *Yue Zongsheng, Xie Jianwen, Zhao Qian, and Meng Deyu*. [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Yue_Semi-Supervised_Video_Deraining_With_Dynamical_Rain_Generator_CVPR_2021_paper.pdf), [[github]](https://github.com/zsyOAOA/S2VD), [[cite]]([Semi-Supervised Video Deraining with Dynamical Rain... - Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Semi-Supervised+Video+Deraining+with+Dynamical+Rain+Generator&btnG=)), 2021.

  2. **Closing the Loop: Joint Rain Generation and Removal via Disentangled Image Translation. (JRGR)** CVPR. ![](https://img.shields.io/badge/single image-purple)

     *Ye yuntong, Chang Yi, Zhou Hanyu, and Yan Luxin*. [[pdf]]([Closing the Loop: Joint Rain Generation and Removal via Disentangled Image Translation (thecvf.com)](https://openaccess.thecvf.com/content/CVPR2021/papers/Ye_Closing_the_Loop_Joint_Rain_Generation_and_Removal_via_Disentangled_CVPR_2021_paper.pdf)), [[cite]]([Closing the Loop: Joint Rain Generation and Removal... - Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Closing+the+Loop%3A+Joint+Rain+Generation+and+Removal+via+Disentangled+Image+Translation&btnG=)), 2021.

  3. **From Rain Generation to Rain Removal. (VRGNet)** CVPR. ![](https://img.shields.io/badge/single image-purple) ![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre train-green) 

     *Wang Hong, Yue Zongsheng, Xie Qi, Zhao Qian, Zheng Yefeng, and Meng Deyu*. [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Ni_Controlling_the_Rain_From_Removal_to_Rendering_CVPR_2021_paper.pdf), [[github]](https://github.com/hongwang01/VRGNet), [[cite]]([From Rain Generation to Rain Removal. - Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=From+Rain+Generation+to+Rain+Removal.+&btnG=)), 2021.

  4. **Controlling the Rain: from Removal to Rendering. (RICNet)** CVPR. ![](https://img.shields.io/badge/single image-purple)

     *Ni Siqi, Cao Xueyun, Yue Tao, and Hu Xuemei*. [[pdf]]([Controlling the Rain: From Removal to Rendering (thecvf.com)](https://openaccess.thecvf.com/content/CVPR2021/papers/Ni_Controlling_the_Rain_From_Removal_to_Rendering_CVPR_2021_paper.pdf)), [[cite]]([Controlling the Rain: from Removal to Rendering - Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Controlling+the+Rain%3A+from+Removal+to+Rendering&btnG=)), 2021.

  5. 

- <h4>Recurrent Model</h4>

1. 

### Prior Based

### Hybrid

### High Level

1. **Beyond Monocular Deraining: Stereo Image Deraining via Semantic Understanding. (PRRNet)** ECCV. ![](https://img.shields.io/badge/single image-purple)

   *Zhang  Kaihao, Luo Wenhan, Ren Wenqi, Wang Jingwen, Zhao Fang, Ma Lin, and Li Hongdong*. [[pdf]]([123720069.pdf (ecva.net)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720069.pdf)), [[cite]]([Beyond Monocular Deraining: Stereo Image Deraining... - Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Beyond+Monocular+Deraining%3A+Stereo+Image+Deraining+via+Semantic+Understanding&btnG=)), 2020.

2. 




## Other Contributors
