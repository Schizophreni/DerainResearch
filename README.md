# Image De-raining papers

![](https://img.shields.io/badge/recent%20update-2022%20Jun.-red) ![](https://img.shields.io/badge/PaperNumber-64-brightgreen) ![](https://img.shields.io/badge/PRs-Welcome-red) ![](https://img.shields.io/badge/Issues-Welcome-red) 

Papers on Image de-raining which include recent prior based and learning based methods. The paper list is mainly maintained by  [Schizophreni](https://github.com/Schizophreni/). We have merged the paper listed in [DerainZoo](https://github.com/nnUyi/DerainZoo) and re-organized recent papers for better comparison and understanding.  Note that this list is also friendly for writing introduction or related work of your academic paper. 

## Contents

- [Image de-raining papers](#derainpapers)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Papers](#papers)
    - [Survey](#Survey)
    - [Learning based](#Learning-Based)
      - [Linear Decompostion](#Linear-Decomposition)
      - [Generative Model](#Generative-Model)
      - [Recurrent Model](#Recurrent-Model)
    - [Prior Based](#Prior-Based)
    - [Hybrid](#Hybrid)
    - [Image de-raining meets high level vision](#High-Level)
  - [Other Contributors](#Other-Contributors)

***News (2022-06-21)***: *add CVPR 2022 image resotration methods*

***News (2022-06-11)***: *add MAXIM (CVPR 2022, multi-task image processing)*

***News (2022-03-07)***: *add DCD-GAN (CVPR 2022, unsup. & constrastive)*


## Introduction

This is a paper list about *image de-raining* researches. Image de-raining focuses on restoring the clean background given the rain-contaminated images as input. The basic assumption for image de-raining is that the information required for recovering the degraded pixels can be extracted from its neighbors.

## Marks

> Task domain: ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/prune/compression-yellow) ![](https://img.shields.io/badge/video-orange) ![](https://img.shields.io/badge/image%20restoration-pink)
>
> Net type: ![](https://img.shields.io/badge/transformer-black)
>
> Resources: ![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre%20train-green)

## Papers

### Survey

1. **A Comprehensive Benchmark Analysis of Single Image Deraining: Current Challenges and Future Perspectives.** IJCV

   *Li Siyuan, Ren Wenqi, Wang Feng, Araujo Iago Breno, E. Tokuda Eric, H. Junior Roberto, M. Cesar-Jr. Roberto, Wang Zhangyang, and Cao Xiaochun.* [[pdf]](https://link.springer.com/article/10.1007/s11263-020-01416-w), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=A+Comprehensive+Benchmark+Analysis+of+Single+Image+Deraining%3A+Current+Challenges+and+Future+Perspectives&btnG=), 2021.

2. **Single Image Deraining: From Model-Based to Data-Driven and Beyond.**  TPAMI. 

   *Yang Wenhan, T. Tan Robby, Wang Shiqi, Fang Yuming, and Liu Jiaying.*  [[pdf](https://arxiv.org/pdf/1912.07150.pdf)], [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Single+Image+Deraining%3A+From+Model-Based+to+Data-Driven+and+Beyond&btnG=), 2020. 

2. **A Survey on Rain Removal from Video and Single Image.** arXiv.

   *Wang Hong, Wu Yichen, Li Minghan, Zhao Qian, and Meng Deyu.* [[pdf]](https://arxiv.org/pdf/1909.08326.pdf) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=A+Survey+on+Rain+Removal+from+Video+and+Single+Image&btnG=), 2019.

   

### Learning Based

#### Linear Decomposition

1. **Dreaming to Prune Image Deraining Networks.** CVPR. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/prune/compression-yellow)

   *Zou Weiqi, Wang Yang, Fu Xueyang, and Cao Yang.* [[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zou_Dreaming_To_Prune_Image_Deraining_Networks_CVPR_2022_paper.pdf), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Dreaming+to+Prune+Image+Deraining+Networks.+&btnG=), 2022.

2. **KNN Local Attention for Image Restoration. (KIT)** CVPR.  ![](https://img.shields.io/badge/image%20restoration-pink)

   *Lee Hunsang, Choi Hyesong, Sohn Kwanghoon, and Min Dongbo.*[[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_KNN_Local_Attention_for_Image_Restoration_CVPR_2022_paper.pdf), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=KNN+Local+Attention+for+Image+Restoration&btnG=), 2022.

3. **All-In-One Image Restoration for Unknown Corruption. (AirNet)** CVPR.  ![](https://img.shields.io/badge/image%20restoration-pink) ![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre%20train-green)

   *Li Boyun, Liu Xiao, Hu Peng, Wu Zhongqin, Lv Jiancheng, and Peng Xi.*[[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_All-in-One_Image_Restoration_for_Unknown_Corruption_CVPR_2022_paper.pdf), [[github]](https://github.com/XLearning-SCU/2022-CVPR-AirNet), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=All-In-One+Image+Restoration+for+Unknown+Corruption&btnG=), 2022.

4. **TransWeather: Transformer-based Restoration of Images Degraded by Adverse Weather Conditions. (TransWeather)** CVPR. ![](https://img.shields.io/badge/image%20restoration-pink) ![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/transformer-black)

   *Valanarasu Jeya Maria Jose, Yasarla Rajeev, and M. Patel Vishal.* [[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Valanarasu_TransWeather_Transformer-Based_Restoration_of_Images_Degraded_by_Adverse_Weather_Conditions_CVPR_2022_paper.pdf), [[github]](https://github.com/jeya-maria-jose/TransWeather), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=TransWeather%3A+Transformer-based+Restoration+of+Images+Degraded+by+Adverse+Weather+Conditions&btnG=), 2022.

5. **Deep Generalized Unfolding Networks for Image Restoration. (DGUNet)** CVPR. ![](https://img.shields.io/badge/image%20restoration-pink) ![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre%20train-green)

   *Mou chong, Wang Qian, and Zhang Jian.*[[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Mou_Deep_Generalized_Unfolding_Networks_for_Image_Restoration_CVPR_2022_paper.pdf), [[github]](https://github.com/MC-E/Deep-Generalized-Unfolding-Networks-for-Image-Restoration), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Deep+Generalized+Unfolding+Networks+for+Image+Restoration&btnG=), 2022.

6. **Uformer: A General U-Shaped Transformer for Image Restoration. (Uformer)** CVPR. ![](https://img.shields.io/badge/image%20restoration-pink) ![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre%20train-green) ![](https://img.shields.io/badge/transformer-black)

   *Wang Zhendong, Cun Xiaodong, Bao Jianmin, Zhou Wengang, Liu Jianzhuang, and Li Houqiang.*[[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Uformer_A_General_U-Shaped_Transformer_for_Image_Restoration_CVPR_2022_paper.pdf), [[github]](https://github.com/ZhendongWang6/Uformer), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Uformer%3A+A+General+U-Shaped+Transformer+for+Image+Restoration&btnG=), 2022.

7. **Restormer: Efficient Transformer for High-Resolution Image Restoration. (Restormer)** CVPR. ![](https://img.shields.io/badge/image%20restoration-pink) ![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre%20train-green) ![](https://img.shields.io/badge/transformer-black)

   *Zamir Syed Waqas, Arora Aditya, Khan Salman, Hayat Munawar.*[[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf), [[github]](https://github.com/swz30/Restormer), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Restormer%3A+Efficient+Transformer+for+High-Resolution+Image+Restoration&btnG=), 2022.

8. **Unsupervised Deraining: Where Contrastive Learning Meets Self-similarity. (NLCL)** CVPR.  ![](https://img.shields.io/badge/single%20image-purple)

   *Ye Yuntong, Yu Changfeng, Chang Yi, Zhu Lin, Zhao, Xi-le, Yan Luxin, and Tian Yonghong.*[[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Ye_Unsupervised_Deraining_Where_Contrastive_Learning_Meets_Self-Similarity_CVPR_2022_paper.pdf), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Unsupervised+Deraining%3A+Where+Contrastive+Learning+Meets+Self-similarity.+%28NLCL%29&btnG=), 2022.

9. **Online-updated High-order Collaborative Networks for Single Image Deraining. (HCNet)** AAAI. ![](https://img.shields.io/badge/single%20image-purple)

   *Wang Cong, Pan Jinshan, and Wu Xiao-Ming.* [[pdf]](https://arxiv.org/pdf/2202.06568.pdf), 2022.

10. **MAXIM: Multi-Axis MLP for Image Processing. ???MAXIM)** CVPR. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre%20train-green)

    *Zhengzhong Tu, Hossein Talebi, Han Zhang, Feng Yang, Peyman Milanar, Alan Bovik, and YinXiao Li*. [[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Tu_MAXIM_Multi-Axis_MLP_for_Image_Processing_CVPR_2022_paper.pdf), [[github]](https://github.com/vztu/maxim-pytorch), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=MAXIM%3A+Multi-Axis+MLP+for+Image+Processing&btnG=), 2022.

11. **Structure-Preserving Deraining with Residue Channel Prior Guidance. (SPDNet)** ICCV. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre%20train-green)

    *Yi Qiaosi, Li Juncheng, Dai Qinyan, Fang Faming, Zhang Guixu, and Zeng Tieyong*. [[pdf]](https://junchenglee.com/paper/ICCV_2021.pdf), [[github]](https://github.com/Joyies/SPDNet), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Structure-Preserving+Deraining+with+Residue+Channel+Prior+Guidance&btnG=), 2021.

12. **Unpaired Learning for Deep Image Deraining With Rain Direction Regularizer** ICCV. ![](https://img.shields.io/badge/single%20image-purple)

    *Liu Yang, Yue Ziyu, Pan Jinshan, and Su Zhixun.* [[pdf]](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Unpaired_Learning_for_Deep_Image_Deraining_With_Rain_Direction_Regularizer_ICCV_2021_paper.pdf), [[github]](https://github.com/Yueziyu/RainDirection-and-Real3000-Dataset), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Unpaired+Learning+for+Deep+Image+Deraining+with+Rain+Direction+Regularizer&btnG=), 2021.

13. **Spatially-Adaptive Image Restoration using Distortion-Guided Networks. (SPAIR)** ICCV. ![](https://img.shields.io/badge/single%20image-purple)

    *Purohit Kuldeep, Suin Maitreya, A.N. Rajagopalan, and Vishnu Naresh Boddeti.* [[pdf]](https://openaccess.thecvf.com/content/ICCV2021/papers/Purohit_Spatially-Adaptive_Image_Restoration_Using_Distortion-Guided_Networks_ICCV_2021_paper.pdf), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Spatially-Adaptive+Image+Restoration+using+Distortion-Guided+Networks&btnG=), 2021.

14. **Improving De-raining Generalization via Neural Reorganization. (NR)** ICCV. ![](https://img.shields.io/badge/single%20image-purple)

    *Xiao Jie, Zhou Man, Fu Xueyang, Liu Aiping, and Zha Zheng-Jun.* [[pdf]](https://openaccess.thecvf.com/content/ICCV2021/papers/Xiao_Improving_De-Raining_Generalization_via_Neural_Reorganization_ICCV_2021_paper.pdf) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=https%3A%2F%2Fopenaccess.thecvf.com%2Fcontent%2FICCV2021%2Fpapers%2FXiao_Improving_De-Raining_Generalization_via_Neural_Reorganization_ICCV_2021_paper.pdf&btnG=#d=gs_cit&u=%2Fscholar%3Fq%3Dinfo%3AH14kj_iZ88cJ%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Den), 2021.

15. **Robust Representation Learning with Feedback for Single Image Deraining. (RLNet)** CVPR. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue)

    *Chen Chenghao, and Li Hao.* [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Robust_Representation_Learning_With_Feedback_for_Single_Image_Deraining_CVPR_2021_paper.pdf) [[github]](https://github.com/LI-Hao-SJTU/DerainRLNet) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Robust+Representation+Learning+with+Feedback+for+Single+Image+Deraining&btnG=), 2021.

16. **Image De-raining via Continual Learning. (PIGWM)** CVPR. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue)

    *Zhou Man, Xiao Jie, Chang Yifan, Fu Xueyang, Liu Aiping, Pan Jinshan, and Zha Zheng-Jun.* [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhou_Image_De-Raining_via_Continual_Learning_CVPR_2021_paper.pdf) [[github]](https://github.com/unpairdenosie/Image-Deraining-via-Continual-Learning) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Image+De-raining+via+Continual+Learning&btnG=), 2021.

17. **Removing Raindrops and Rain Streaks in One Go. (CCN)** CVPR. ![](https://img.shields.io/badge/single%20image-purple)

    *Quan Ruijie, Yu Xin, Liang Yuanzhi, and Yang Yi*. [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Quan_Removing_Raindrops_and_Rain_Streaks_in_One_Go_CVPR_2021_paper.pdf), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=**Removing+Raindrops+and+Rain+Streaks+in+One+Go&btnG=), 2021.

18. **Multi-Stage Progressive Image Restoration. (MPRNet)** CVPR. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre%20train-green)

    *Zamir Syed Waqas, Arora Aditya, Khan Salman, Hayat Munawar, Khan Fahad Shabaz, Yang Ming-Hsuan, and Shao Ling.* [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Zamir_Multi-Stage_Progressive_Image_Restoration_CVPR_2021_paper.pdf), [[github]](https://github.com/swz30/MPRNet), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Multi-Stage+Progressive+Image+Restoration&btnG=), 2021.

19. **Rain Streak Removal via Dual Graph Convolutional Network. (DualGCN)** AAAI. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue)

    *Fu Xueyang, Qi Qi, Zha Zheng-Jun, Zhu Yurui, and Ding Xinghao*. [[pdf]](https://www.aaai.org/AAAI21Papers/AAAI-228.FuXY.pdf), [[github]](https://xueyangfu.github.io/paper/2021/AAAI/code.zip), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Rain+Streak+Removal+via+Dual+Graph+Convolutional+Network.&btnG=), 2021. 

20. **Pre-Trained Image Processing Transformer. (IPT)** CVPR. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/transformer-black)

    *Chen Hanting, Wang Yunhe, Guo Tianyu, Xu Chang, Deng Yipeng, Liu Zhenhua, Ma Siwei, Xu Chunjing, Xu Chao, and Gao Wen*. [[pdf]](https://arxiv.org/pdf/2012.00364.pdf), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Pre-Trained+Image+Processing+Transformer&btnG=), 2021.

21. **Unpaired Adversarial Learning for Single Image Deraining with Rain-Space Contrastive Constraints. (CDR-GAN)** arXiv. ![](https://img.shields.io/badge/single%20image-purple)

    *Chen Xiang, Pan Jinshan, Jiang Kui, Huang Yufeng, Kong Caihua, Dai Longgang, and Li Yufeng.*[[pdf]](https://arxiv.org/abs/2109.02973), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Unpaired+adversarial+learning+for+single+image+deraining+with+rain-space+contrastive+constraints.&btnG=#d=gs_cit&u=%2Fscholar%3Fq%3Dinfo%3Am1qUPIuQEkcJ%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Den), 2021.

22. **SDNET: Multi-Branch for Single Image Deraining Using Swin. (SDNet).** arXiv. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/transformer-black)

     *Tan Fuxiang, Kong Yuting, Fan Yingying, Liu Feng, Zhou Daxin, Zhang Hao, Chen Long, and Gao Liang.* [[pdf]](https://arxiv.org/pdf/2105.15077.pdf), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=SDNET%3A+Multi-branch+for+single+image+deraining+using+swin&btnG=), 2021.

23. **Rain Removal and Illumination Enhancement Done in One Go. (EMNet)** arXiv. ![](https://img.shields.io/badge/single%20image-purple)

     *Wan Yecong, Cheng Yuanshuo, and Shao Mingwen.*[[pdf]](https://arxiv.org/abs/2108.03873), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Rain+Removal+and+Illumination+Enhancement+Done+in+One+Go&btnG=), 2021.

24. **Blind Image Decomposition. (BID)** arXiv. ![](https://img.shields.io/badge/single%20image-purple)

      *Han Junlin, Li Weihao, Fang Pengfei, Sun Chunyi, Hong Jie, Mohammad Ali Armin, Lars Petersson, and Li Hongdong.* [[pdf]](https://arxiv.org/abs/2108.11364), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Blind+image+decomposition&btnG=), 2021.

25. **Self-Learning Video Rain Streak Removal: When Cyclic Consistency Meets Temporal Correspondence. (SLDNet)** CVPR. ![](https://img.shields.io/badge/video-orange)

     *Yang Wenhan, T. Tan Robby, Wang Shiqi, and Liu Jiaying.*[[pdf]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Self-Learning_Video_Rain_Streak_Removal_When_Cyclic_Consistency_Meets_Temporal_CVPR_2020_paper.pdf) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Self-Learning+Video+Rain+Streak+Removal%3A+When+Cyclic+Consistency+Meets+Temporal+Correspondence&btnG=), 2020.

26. **All in One Bad Weather Removal using Architectural Search. (NAS)** CVPR. ![](https://img.shields.io/badge/single%20image-purple)

     *Li Ruoteng, T. Tan Robby, and Cheong Looeng-Fah.*[[pdf]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_All_in_One_Bad_Weather_Removal_Using_Architectural_Search_CVPR_2020_paper.pdf) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=All+in+One+Bad+Weather+Removal+using+Architectural+Search&btnG=), 2020.

27. **Wavelet-based dual-branch network for image demoir??ing. (WDNet)** ECCV. ![](https://img.shields.io/badge/single%20image-purple)

      *Liu Lin, Liu Jianzhuang, Yuan Shanxin, Slabaugh Gregory, Leonardis Ales, Zhou Wengang, and Tian Qi.*[[pdf]](https://link.springer.com/chapter/10.1007%2F978-3-030-58601-0_6) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Wavelet-based+dual-branch+network+for+image+demoir%C3%A9ing&btnG=), 2020.

28. **Rethinking Image Deraining via Rain Streaks and Vapors. (S-V-ANet)** ECCV. ![](https://img.shields.io/badge/single%20image-purple)

       *Wang Yinglong, Song Yibing, Ma Chao, and Zeng Bing.* [[pdf]](https://arxiv.org/pdf/2008.00823.pdf) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Rethinking+Image+Deraining+via+Rain+Streaks+and+Vapors&btnG=), 2020.

29. **Joint Self-Attention and Scale-Aggregation for Self-Calibrated Deraining Network. (JDNet)** ACM'MM. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre%20train-green)

       *Wang Cong, Wu Yutong, Su Zhixun, and Chen Junyang.* [[pdf]](https://arxiv.org/pdf/2008.02763.pdf) [[github]](https://github.com/Ohraincu/JDNet) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Joint+Self-Attention+and+Scale-Aggregation+for+Self-Calibrated+Deraining+Network&btnG=), 2020.

30. **DCSFN: Deep Cross-scale Fusion Network for Single Image Rain Removal. (DCSFN)** ACM'MM. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre%20train-green)

      *Wang Cong, Xing Xiaoying, Su Zhixun, and Chen Junyang.*[[pdf]](https://arxiv.org/pdf/2008.00767.pdf) [[github]](https://github.com/Ohraincu/DCSFN) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=DCSFN%3A+Deep+Cross-scale+Fusion+Network+for+Single+Image+Rain+Removal.&btnG=), 2020.

31. **Conditional Variational Image Deraining. (CVID)** TIP. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue)

      *Du Yingjun, Xu Jun, Zhen Xiantong, Cheng Ming-Ming, and Shao Ling.* [[pdf]](file:///D:/Education/Papers/derain/conditional_variational_image_deraining.pdf) [[github]](https://github.com/Yingjun-Du/VID) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Conditional+variational+image+deraining&btnG=), 2020.

32. **Variational Image Deraining. (VID)** WACV. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue)

      *Du Yingjun, Xun Jun, Qiu Qiang, Zhen Xiantong, and Zhang Lei.*[[pdf]](https://openaccess.thecvf.com/content_WACV_2020/papers/Du_Variational_Image_Deraining_WACV_2020_paper.pdf) [[github]](https://github.com/Yingjun-Du/VID) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=variational+image+deraining&btnG=&oq=Variational+Image+Derai), 2020.

33. **Detail-recovery Image Deraining via Context Aggregation Networks. (DRD-Net)** CVPR. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue)

      *Deng Sen, Wei Mingqiang, Wang Jun, Feng Yidan, Liang Luming, Xie Haoran, Wang Fu Lee, and Wang Meng.* [[pdf]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Deng_Detail-recovery_Image_Deraining_via_Context_Aggregation_Networks_CVPR_2020_paper.pdf) [[github]](https://github.com/Dengsgithub/DRD-Net) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Detail-recovery+Image+Deraining+via+Context+Aggregation+Networks&btnG=), 2020.

34. **Physical Model Guided Deep Image Deraining.** ICME. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue)

      *Zhu Honghe, Wang Cong, Zhang Yajie, Su Zhixun, and Zhao Guohui.* [[pdf]](https://arxiv.org/pdf/2003.13242.pdf) [[github]](https://github.com/Ohraincu/PHYSICAL-MODEL-GUIDED-DEEP-IMAGE-DERAINING) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Physical+Model+Guided+Deep+Image+Deraining.&btnG=), 2020.

35. **RDDAN: A Residual Dense Dilated Aggregated Network for Single Image Deraining. (RDDAN)** ICME. ![](https://img.shields.io/badge/single%20image-purple)

      *Yang Youzhao, Ran Wu, and Lu Hong.* [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9102945) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=RDDAN%3A+A+Residual+Dense+Dilated+Aggregated+Network+for+Single+Image+Deraining.&btnG=), 2020.

36. **Confidence Measure Guided Single Image De-Raining. (QuDec)** TIP. ![](https://img.shields.io/badge/single%20image-purple)

       *Yasarla Rajeev, and M. Patel Vishal.* [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9007569) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Confidence+Measure+Guided+Single+Image+De-Raining&btnG=), 2020.

37. **A Coarse-to-Fine Multi-stream Hybrid Deraining Network for Single Image Deraining. (MH-DerainNet)** ICDM. ![](https://img.shields.io/badge/single%20image-purple)

      *Wei Yanyan, Zhang Zhao, Zhang Haijun, Hong Richang, and Wang Meng.* [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8970838) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=A+Coarse-to-Fine+Multi-stream+Hybrid+Deraining+Network+for+Single+Image+Deraining&btnG=), 2019.

38. **ERL-Net: Entangled Representation Learning for Single Image De-Raining. (ERL-Net)** ICCV. ![](https://img.shields.io/badge/single%20image-purple)

      *Wang Guoqing, Sun Changming, and Sowmya Acrot.* [[pdf]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_ERL-Net_Entangled_Representation_Learning_for_Single_Image_De-Raining_ICCV_2019_paper.pdf) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=ERL-Net%3A+Entangled+Representation+Learning+for+Single+Image+De-Raining&btnG=), 2019.

39. **DTDN: Dual-task de-raining network. (DTDN)** ACM'MM. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre%20train-green)

      *Wang Zheng, Li Jianwu, and Song Ge.* [[pdf]](https://arxiv.org/pdf/2008.09326.pdf), [[github]](https://github.com/long-username/DTDN-DTDN-Dual-task-De-raining-Network), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=DTDN%3A+Dual-task+Training+Network&btnG=), 2019.

40. **Gradual Network for Single Image De-raining. (GraNet)** ACM'MM. ![](https://img.shields.io/badge/single%20image-purple)

      *Yu Weijiang, Huang Zhe, Zhang Wayne, Feng Litong, and Xiao Nong.* [[pdf]](https://arxiv.org/pdf/1909.09677.pdf), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Gradual+Network+for+Single+Image+De-raining&btnG=), 2019.

41. **An Effective Two-Branch Model-Based Deep Network for Single Image Deraining. (AMPE-Net)** arXiv. ![](https://img.shields.io/badge/single%20image-purple)

    *Wang Yinglong, Gong Dong, Yang Jie, Shi Qinfeng, Anton van den Hengel, Xie Dehua, and Zeng Bing.* [[pdf]](https://arxiv.org/pdf/1905.05404.pdf), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=An+Effective+Two-Branch+Model-Based+Deep+Network+for+Single+Image+Deraining&btnG=#d=gs_cit&u=%2Fscholar%3Fq%3Dinfo%3AN5oEFLEcteoJ%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Den), 2019.
    ****

#### Generative Model

1. **Unpaired Deep Image Deraining Using Dual Contrastive Learning. (DCD-GAN)** CVPR. ![](https://img.shields.io/badge/single%20image-purple)

   *Chen Xiang, Pan Jinshan, Jiang Kui, Li Yufeng, Huang Yufeng, Kong Caihua, Dai Longgang, and Fan Zhentao.* [[homepage]](https://cxtalk.github.io/projects/DCD-GAN.html#). 

2. **Semi-Supervised Video Deraining with Dynamical Rain Generator. (S2VD)** CVPR. ![](https://img.shields.io/badge/video-orange) ![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre%20train-green) 

   *Yue Zongsheng, Xie Jianwen, Zhao Qian, and Meng Deyu*. [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Yue_Semi-Supervised_Video_Deraining_With_Dynamical_Rain_Generator_CVPR_2021_paper.pdf), [[github]](https://github.com/zsyOAOA/S2VD), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Semi-Supervised+Video+Deraining+with+Dynamical+Rain+Generator&btnG=), 2021.

3. **Closing the Loop: Joint Rain Generation and Removal via Disentangled Image Translation. (JRGR)** CVPR. ![](https://img.shields.io/badge/single%20image-purple)

   *Ye yuntong, Chang Yi, Zhou Hanyu, and Yan Luxin*. [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Ye_Closing_the_Loop_Joint_Rain_Generation_and_Removal_via_Disentangled_CVPR_2021_paper.pdf), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Closing+the+Loop%3A+Joint+Rain+Generation+and+Removal+via+Disentangled+Image+Translation&btnG=), 2021.

4. **From Rain Generation to Rain Removal. (VRGNet)** CVPR. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre%20train-green) 

   *Wang Hong, Yue Zongsheng, Xie Qi, Zhao Qian, Zheng Yefeng, and Meng Deyu*. [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Ni_Controlling_the_Rain_From_Removal_to_Rendering_CVPR_2021_paper.pdf), [[github]](https://github.com/hongwang01/VRGNet), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=From+Rain+Generation+to+Rain+Removal.+&btnG=), 2021.

5. **Controlling the Rain: from Removal to Rendering. (RICNet)** CVPR. ![](https://img.shields.io/badge/single%20image-purple)

   *Ni Siqi, Cao Xueyun, Yue Tao, and Hu Xuemei*. [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Ni_Controlling_the_Rain_From_Removal_to_Rendering_CVPR_2021_paper.pdf), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Controlling+the+Rain%3A+from+Removal+to+Rendering&btnG=), 2021.


#### Recurrent Model

1. **Towards Robust Rain Removal Against Adversarial Attacks: A Comprehensive Benchmark Analysis and Beyond. ** CVPR.  ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre%20train-green)

   *Yu Li, Yang Wenhan, Tan Yap-Peng, and C. Kot Alex.*[[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_Towards_Robust_Rain_Removal_Against_Adversarial_Attacks_A_Comprehensive_Benchmark_CVPR_2022_paper.pdf), [[github]](https://github.com/yuyi-sd/Robust_Rain_Removal), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Towards+Robust+Rain+Removal+Against+Adversarial+Attacks%3A+A+Comprehensive+Benchmark+Analysis+and+Beyond&btnG=), 2022.

2. **Multi-Scale Progressive Fusion Network for Single Image Deraining.** CVPR. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre%20train-green)

   *Jiang Kui, Wang Zhongyuan, Yi Peng, and Chen Chen.* [[pdf]](https://arxiv.org/pdf/2003.10985.pdf) [[github]](https://github.com/kuijiang0802/MSPFN) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Multi-Scale+Progressive+Fusion+Network+for+Single+Image+Deraining.&btnG=), 2020.

3. **Single Image Deraining Using Bilateral Recurrent Network. (BRN)** TIP. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre%20train-green)

   *Ren Dongwei, Shang wei, Zhu Pengfei, Hu Qinghua, Meng Deyu, and Zuo Wangmeng.* [[pdf]](https://csdwren.github.io/papers/2020_tip_BRN.pdf), [[github]](https://github.com/csdwren/RecDerain), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Single+Image+Deraining+Using+Bilateral+Recurrent+Network&btnG=), 2020. 

4. **Single Image Deraining via Recurrent Hierarchy and Enhancement Network. (ReHEN)** ACM'MM. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre%20train-green)

   *Yang Youzhao, and Lu Hong.* [[pdf]](https://dl.acm.org/doi/10.1145/3343031.3351149#URLTOKEN#)  [[github]](https://github.com/nnUyi/ReHEN) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Single+Image+Deraining+via+Recurrent+Hierarchy+Enhancement+Network&btnG=), 2019.

5. **Single Image Deraining using a Recurrent Multi-scale Aggregation and Enhancement Network. (ReMAEN)** ICME. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/pre%20train-green)  

   *Yang Youzhao, and Lu Hong.* [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8784948) [[github]](https://github.com/nnUyi/ReMAEN) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Single+Image+Deraining+using+a+Recurrent+Multi-scale+Aggregation+and+Enhancement+Network&btnG=), 2019.



### Prior Based

1. **Single Image Rain Removal Boosting via Directional Gradient. (DiG-CoM)** ICME. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue)

   *Ran Wu, Yang Youzhao, Lu Hong.* [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9102800) [[github]](https://github.com/Schizophreni/Set-vanish-to-the-rain) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Single+Image+Rain+Removal+Boosting+via+Directional+Gradient&btnG=), 2020.

### Hybrid

1. **Unsupervised Image Deraining Optimization Model Driven Deep CNN. (UDGNet)** ACM'MM. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre%20train-green)

   *Yu Changfeng, Chang Yi, Li Yi, Zhao Xile, and Yan Luxin*. [[pdf]](https://owuchangyuo.github.io/files/UDGNet.pdf), [[github]](https://github.com/ChangfengYu-Hust/UDGNet), [[cite]](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=Unsupervised+Image+Deraining+Optimization+Model+Driven+Deep+CNN&btnG=), 2021.

2. **A Model-driven Deep Neural Network for Single Image Rain Removal. (RCDNet)** CVPR.  ![](https://img.shields.io/badge/single%20image-purple)![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre%20train-green)

   *Wang Hong, Xie Qi, Zhao Qian, and Meng Deyu.* [[pdf]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_A_Model-Driven_Deep_Neural_Network_for_Single_Image_Rain_Removal_CVPR_2020_paper.pdf) [[github]](https://github.com/hongwang01/RCDNet) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=A+Model-driven+Deep+Neural+Network+for+Single+Image+Rain+Removal&btnG=), 2020.

3. **Syn2Real Transfer Learning for Image Deraining using Gaussian Processes. (Syn2Real)** CVPR. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue) ![](https://img.shields.io/badge/pre%20train-green)

   *Yasarla Rajeev, A. Sindagi Vishwanath, and M. Patel Vishal.* [[pdf]](https://arxiv.org/pdf/2006.05580.pdf) [[github]](https://github.com/rajeevyasarla/Syn2Real) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Syn2Real+Transfer+Learning+for+Image+Deraining+using+Gaussian+Processes.+&btnG=), 2020.

4. **Scale-Free Single Image Deraining Via VisibilityEnhanced Recurrent Wavelet Learning. (RWL)** TIP. ![](https://img.shields.io/badge/single%20image-purple)

   *Yang Wenhan, Liu Jiaying, Yang Shuai, and Guo Zongming.* [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8610325) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Scale-Free+Single+Image+Deraining+Via+Visibility%02Enhanced+Recurrent+Wavelet+Learnin&btnG=), 2019.

### High Level

1. **Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions** arXiv. ![](https://img.shields.io/badge/single%20image-purple)![](https://img.shields.io/badge/project-blue)![](https://img.shields.io/badge/pre%20train-green)

   *Liu Wenyu, Ren Gaofeng, Yu Runsheng, Guo Shi, Zhu Jianke, and Zhang Lei.* [[pdf]](https://arxiv.org/pdf/2112.08088.pdf), [[github]](https://github.com/wenyyu/Image-Adaptive-YOLO), [[cite]](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=Image-Adaptive+YOLO+for+Object+Detection+in+Adverse+Weather+Conditions&btnG=), 2022. 

2. **RaidaR: a rich annotated image dataset of rainy street scenes. (RaidaR)** arXiv. ![](https://img.shields.io/badge/single%20image-purple)

   *Jin Jiongchao, Fatemi Arezou, Lira Wallace, Yu Fenggen, Leng Biao, Ma Rui, Ali Mahdavi-Amiri, and Zhang Hao.* [[pdf]](https://arxiv.org/abs/2104.04606), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=RaidaR%3A+a+rich+annotated+image+dataset+of+rainy+street+scenes&btnG=), 2021.

3. **Beyond Monocular Deraining: Stereo Image Deraining via Semantic Understanding. (PRRNet)** ECCV. ![](https://img.shields.io/badge/single%20image-purple)

   *Zhang  Kaihao, Luo Wenhan, Ren Wenqi, Wang Jingwen, Zhao Fang, Ma Lin, and Li Hongdong*. [[pdf]]([123720069.pdf (ecva.net)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720069.pdf)), [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Beyond+Monocular+Deraining%3A+Stereo+Image+Deraining+via+Semantic+Understanding&btnG=), 2020.

4. **ForkGAN: Seeing into the Rainy Night. (ForkGAN)** ECCV. ![](https://img.shields.io/badge/single%20image-purple) ![](https://img.shields.io/badge/project-blue)

   *Zheng Ziqiang, Wu Yang, Han Xinran, and Shi Jianbo.*[[pdf]](https://link.springer.com/chapter/10.1007%2F978-3-030-58580-8_10) [[github]](https://github.com/zhengziqiang/ForkGAN) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=**ForkGAN%3A+Seeing+into+the+Rainy+Night&btnG=), 2020.

5. **RainFlow: Optical Flow under Rain Streaks and Rain Veiling Effect. (RainFlow)** ICCV. ![](https://img.shields.io/badge/single%20image-purple)

   *Li Ruoteng, T. Tan Robby, Cheong Loong-Fah, I. Aviles-Rivero Angelica, Fan Qingnan, and Schonlieb Carola-Bibiane.* [[pdf]](https://fqnchina.github.io/QingnanFan_files/iccv_2019.pdf) [[cite]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=RainFlow%3A+Optical+Flow+under+Rain+Streaks+and+Rain+Veiling+Effect&btnG=), 2019.


## Other Contributors
