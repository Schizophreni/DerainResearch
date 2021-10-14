# Image De-raining papers

![](https://img.shields.io/badge/recent%20update-2021%20Oct.-red) ![](https://img.shields.io/badge/PaperNumber-27-brightgreen) ![](https://img.shields.io/badge/PRs-Welcome-red) ![](https://img.shields.io/badge/Issues-Welcome-red) 

Must-read papers on Image de-raining which include recent prior based and learning based methods. The paper list is mainly maintained by  [Schizophreni](https://github.com/Schizophreni/). We have merged the paper listed in [DerainZoo](https://github.com/nnUyi/DerainZoo) and re-organized recent papers for better comparison and understanding.  Note that this list is also friendly for writing introduction or related work of your academic paper. 

## Contents

- [Image de-raining papers](#derainpapers)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Papers](#papers)
    - [Survey](#Survey)
    - [Learning based](#Learning%20Based)
      - [Linear Decompostion](#Linear%20Decomposition)
      - [Generation Model](#Generation%20Model)
      - [Recurrent Model](#Recurrent%20Model)
    - [Prior Based](#Prior%20Based)
    - [Hybrid](#Hybrid)
    - [Image de-raining meets high level vision](#High%20Level)
  - [Other Contributors](#other-contributors)



## Introduction

This is a paper list about *image de-raining* researches. Image de-raining focuses on restoring the clean background given the rain-contaminated images as input. The basic assumption for image de-raining is that the information required for recovering the degraded pixels can be extracted from its neighbors.

## Papers

### Survey

1. **Single Image Deraining: From Model-Based to Data-Driven and Beyond.**  TPAMI. 

   *Yang Wenhan, T. Tan Robby, Wang Shiqi, Fang Yuming, and Liu Jiaying.*  [[pdf](https://arxiv.org/pdf/1912.07150.pdf)], [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Single+Image+Deraining%3A+From+Model-Based+to+Data-Driven+and+Beyond&btnG=), 2021. 

### Learning Based

#### Linear Decomposition

1. **Structure-Preserving Deraining with Residue Channel Prior Guidance. (SPDNet)** ICCV. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre%20train-green)

   *Yi Qiaosi, Li Juncheng, Dai Qinyan, Fang Faming, Zhang Guixu, and Zeng Tieyong*. [[pdf]](https://junchenglee.com/paper/ICCV_2021.pdf), [[github]](https://github.com/Joyies/SPDNet), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Structure-Preserving+Deraining+with+Residue+Channel+Prior+Guidance&btnG=), 2021.

2. **Image De-raining via Continual Learning. (PIGWN)** CVPR. ![](https://img.shields.io/badge/single%20image-purple)

   *Zhou Man, Xiao Jie, Chang Yifan, Fu Xueyang, Liu Aiping, Pan Jinshan, and Zha Zheng-Jun.* [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhou_Image_De-Raining_via_Continual_Learning_CVPR_2021_paper.pdf) [[cite]]()

   

3. **Removing Raindrops and Rain Streaks in One Go. (CCN)** CVPR. ![](https://img.shields.io/badge/single%20image-purple)

   *Quan Ruijie, Yu Xin, Liang Yuanzhi, and Yang Yi*. [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Quan_Removing_Raindrops_and_Rain_Streaks_in_One_Go_CVPR_2021_paper.pdf), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=**Removing+Raindrops+and+Rain+Streaks+in+One+Go&btnG=), 2021.

4. **Rain Streak Removal via Dual Graph Convolutional Network. (DualGCN)** AAAI. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue)

   *Fu Xueyang, Qi Qi, Zha Zheng-Jun, Zhu Yurui, and Ding Xinghao*. [[pdf]](https://www.aaai.org/AAAI21Papers/AAAI-228.FuXY.pdf), [[github]](https://xueyangfu.github.io/paper/2021/AAAI/code.zip), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Rain+Streak+Removal+via+Dual+Graph+Convolutional+Network.&btnG=), 2021. 

5. **Pre-Trained Image Processing Transformer. (IPT)** CVPR. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/transformer-black)

   *Chen Hanting, Wang Yunhe, Guo Tianyu, Xu Chang, Deng Yipeng, Liu Zhenhua, Ma Siwei, Xu Chunjing, Xu Chao, and Gao Wen*. [[pdf]]([2012.00364.pdf (arxiv.org)](https://arxiv.org/pdf/2012.00364.pdf)), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Pre-Trained+Image+Processing+Transformer&btnG=), 2021.

6. **Self-Learning Video Rain Streak Removal: When Cyclic Consistency Meets Temporal Correspondence. (SLDNet)** CVPR. ![](https://img.shields.io/badge/video-orange)

   *Yang Wenhan, T. Tan Robby, Wang Shiqi, and Liu Jiaying.*[[pdf]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Self-Learning_Video_Rain_Streak_Removal_When_Cyclic_Consistency_Meets_Temporal_CVPR_2020_paper.pdf) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Self-Learning+Video+Rain+Streak+Removal%3A+When+Cyclic+Consistency+Meets+Temporal+Correspondence&btnG=), 2020.

7. **All in One Bad Weather Removal using Architectural Search. (NAS)** CVPR. ![](https://img.shields.io/badge/single%20image-purple)

   *Li Ruoteng, T. Tan Robby, and Cheong Looeng-Fah.*[[pdf]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_All_in_One_Bad_Weather_Removal_Using_Architectural_Search_CVPR_2020_paper.pdf) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=All+in+One+Bad+Weather+Removal+using+Architectural+Search&btnG=), 2020.

8.  **Wavelet-based dual-branch network for image demoiréing. (WDNet)** ECCV. ![](https://img.shields.io/badge/single%20image-purple)

     *Liu Lin, Liu Jianzhuang, Yuan Shanxin, Slabaugh Gregory, Leonardis Ales, Zhou Wengang, and Tian Qi.*[[pdf]](https://link.springer.com/chapter/10.1007%2F978-3-030-58601-0_6) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Wavelet-based+dual-branch+network+for+image+demoir%C3%A9ing&btnG=), 2020.

9.  **Rethinking Image Deraining via Rain Streaks and Vapors. (S-V-ANet)** ECCV. ![](https://img.shields.io/badge/single%20image-purple)

     *Wang Yinglong, Song Yibing, Ma Chao, and Zeng Bing.* [[pdf]](https://arxiv.org/pdf/2008.00823.pdf) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Rethinking+Image+Deraining+via+Rain+Streaks+and+Vapors&btnG=), 2020.

10. **Joint Self-Attention and Scale-Aggregation for Self-Calibrated Deraining Network. (JDNet)** ACM'MM. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre%20train-green)

     *Wang Cong, Wu Yutong, Su Zhixun, and Chen Junyang.* [[pdf]](https://arxiv.org/pdf/2008.02763.pdf) [[github]](https://github.com/Ohraincu/JDNet) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Joint+Self-Attention+and+Scale-Aggregation+for+Self-Calibrated+Deraining+Network&btnG=), 2020.

11. **DCSFN: Deep Cross-scale Fusion Network for Single Image Rain Removal. (DCSFN)** ACM'MM. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre%20train-green)

    *Wang Cong, Xing Xiaoying, Su Zhixun, and Chen Junyang.*[[pdf]](https://arxiv.org/pdf/2008.00767.pdf) [[github]](https://github.com/Ohraincu/DCSFN) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=DCSFN%3A+Deep+Cross-scale+Fusion+Network+for+Single+Image+Rain+Removal.&btnG=), 2020.

12. **Conditional Variational Image Deraining. (CVID)** TIP. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue)

    *Du Yingjun, Xu Jun, Zhen Xiantong, Cheng Ming-Ming, and Shao Ling.* [[pdf]](file:///D:/Education/Papers/derain/conditional_variational_image_deraining.pdf) [[github]](https://github.com/Yingjun-Du/VID) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Conditional+variational+image+deraining&btnG=), 2020.

13. **Variational Image Deraining. (VID)** WACV. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue)

    *Du Yingjun, Xun Jun, Qiu Qiang, Zhen Xiantong, and Zhang Lei.*[[pdf]](https://openaccess.thecvf.com/content_WACV_2020/papers/Du_Variational_Image_Deraining_WACV_2020_paper.pdf) [[github]](https://github.com/Yingjun-Du/VID) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=variational+image+deraining&btnG=&oq=Variational+Image+Derai), 2020.

14.  **Detail-recovery Image Deraining via Context Aggregation Networks. (DRD-Net)** CVPR. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue)

     *Deng Sen, Wei Mingqiang, Wang Jun, Feng Yidan, Liang Luming, Xie Haoran, Wang Fu Lee, and Wang Meng.* [[pdf]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Deng_Detail-recovery_Image_Deraining_via_Context_Aggregation_Networks_CVPR_2020_paper.pdf) [[github]](https://github.com/Dengsgithub/DRD-Net) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Detail-recovery+Image+Deraining+via+Context+Aggregation+Networks&btnG=), 2020.

15.  **Multi-Scale Progressive Fusion Network for Single Image Deraining.** CVPR. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre%20train-green)

     *Jiang Kui, Wang Zhongyuan, Yi Peng, and Chen Chen.* [[pdf]](https://arxiv.org/pdf/2003.10985.pdf) [[github]](https://github.com/kuijiang0802/MSPFN) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Multi-Scale+Progressive+Fusion+Network+for+Single+Image+Deraining.&btnG=), 2020.

16. **Physical Model Guided Deep Image Deraining.** ICME. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue)

    *Zhu Honghe, Wang Cong, Zhang Yajie, Su Zhixun, and Zhao Guohui.* [[pdf]](https://arxiv.org/pdf/2003.13242.pdf) [[github]](https://github.com/Ohraincu/PHYSICAL-MODEL-GUIDED-DEEP-IMAGE-DERAINING) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Physical+Model+Guided+Deep+Image+Deraining.&btnG=), 2020.

17. **RDDAN: A Residual Dense Dilated Aggregated Network for Single Image Deraining. (RDDAN)** ICME. ![](https://img.shields.io/badge/single%20image-purple)

    *Yang Youzhao, Ran Wu, and Lu Hong.* [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9102945) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=RDDAN%3A+A+Residual+Dense+Dilated+Aggregated+Network+for+Single+Image+Deraining.&btnG=), 2020.

18. **Confidence Measure Guided Single Image De-Raining. (QuDec)** TIP. ![](https://img.shields.io/badge/single%20image-purple)

    *Yasarla Rajeev, and M. Patel Vishal.* [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9007569) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Confidence+Measure+Guided+Single+Image+De-Raining&btnG=), 2020.

    ****

#### Generation Model

1. **Semi-Supervised Video Deraining with Dynamical Rain Generator. (S2VD) ** CVPR. ![](https://img.shields.io/badge/video-orange) ![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre%20train-green) 

   *Yue Zongsheng, Xie Jianwen, Zhao Qian, and Meng Deyu*. [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Yue_Semi-Supervised_Video_Deraining_With_Dynamical_Rain_Generator_CVPR_2021_paper.pdf), [[github]](https://github.com/zsyOAOA/S2VD), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Semi-Supervised+Video+Deraining+with+Dynamical+Rain+Generator&btnG=), 2021.

2. **Closing the Loop: Joint Rain Generation and Removal via Disentangled Image Translation. (JRGR)** CVPR. ![](https://img.shields.io/badge/single%20image-purple)

   *Ye yuntong, Chang Yi, Zhou Hanyu, and Yan Luxin*. [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Ye_Closing_the_Loop_Joint_Rain_Generation_and_Removal_via_Disentangled_CVPR_2021_paper.pdf), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Closing+the+Loop%3A+Joint+Rain+Generation+and+Removal+via+Disentangled+Image+Translation&btnG=), 2021.

3. **From Rain Generation to Rain Removal. (VRGNet)** CVPR. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre%20train-green) 

   *Wang Hong, Yue Zongsheng, Xie Qi, Zhao Qian, Zheng Yefeng, and Meng Deyu*. [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Ni_Controlling_the_Rain_From_Removal_to_Rendering_CVPR_2021_paper.pdf), [[github]](https://github.com/hongwang01/VRGNet), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=From+Rain+Generation+to+Rain+Removal.+&btnG=), 2021.

4. **Controlling the Rain: from Removal to Rendering. (RICNet)** CVPR. ![](https://img.shields.io/badge/single%20image-purple)

   *Ni Siqi, Cao Xueyun, Yue Tao, and Hu Xuemei*. [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Ni_Controlling_the_Rain_From_Removal_to_Rendering_CVPR_2021_paper.pdf), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Controlling+the+Rain%3A+from+Removal+to+Rendering&btnG=), 2021.

5. 

#### Recurrent Model

1. 

### Prior Based

1. **Single Image Rain Removal Boosting via Directional Gradient. (DiG-CoM)** ICME. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue)

   *Ran Wu, Yang Youzhao, Lu Hong.* [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9102800) [[github]](https://github.com/Schizophreni/Set-vanish-to-the-rain) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Single+Image+Rain+Removal+Boosting+via+Directional+Gradient&btnG=), 2020.

2. 

### Hybrid

1. **A Model-driven Deep Neural Network for Single Image Rain Removal. (RCDNet)** CVPR.  ![](https://img.shields.io/badge/single%20image-purple)![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre%20train-green)

   *Wang Hong, Xie Qi, Zhao Qian, and Meng Deyu.* [[pdf]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_A_Model-Driven_Deep_Neural_Network_for_Single_Image_Rain_Removal_CVPR_2020_paper.pdf) [[github]](https://github.com/hongwang01/RCDNet) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=A+Model-driven+Deep+Neural+Network+for+Single+Image+Rain+Removal&btnG=), 2020.

2. **Syn2Real Transfer Learning for Image Deraining using Gaussian Processes. (Syn2Real)** CVPR. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre%20train-green)

   *Yasarla Rajeev, A. Sindagi Vishwanath, and M. Patel Vishal.* [[pdf]](https://arxiv.org/pdf/2006.05580.pdf) [[github]](https://github.com/rajeevyasarla/Syn2Real) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Syn2Real+Transfer+Learning+for+Image+Deraining+using+Gaussian+Processes.+&btnG=), 2020.

3. 

### High Level

1. **Beyond Monocular Deraining: Stereo Image Deraining via Semantic Understanding. (PRRNet)** ECCV. ![](https://img.shields.io/badge/single%20image-purple)

   *Zhang  Kaihao, Luo Wenhan, Ren Wenqi, Wang Jingwen, Zhao Fang, Ma Lin, and Li Hongdong*. [[pdf]]([123720069.pdf (ecva.net)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720069.pdf)), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Beyond+Monocular+Deraining%3A+Stereo+Image+Deraining+via+Semantic+Understanding&btnG=), 2020.

2. **ForkGAN: Seeing into the Rainy Night. (ForkGAN)** ECCV. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue)

   *Zheng Ziqiang, Wu Yang, Han Xinran, and Shi Jianbo.*[[pdf]](https://link.springer.com/chapter/10.1007%2F978-3-030-58580-8_10) [[github]](https://github.com/zhengziqiang/ForkGAN) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=**ForkGAN%3A+Seeing+into+the+Rainy+Night&btnG=), 2020.

3.  


## Other Contributors
