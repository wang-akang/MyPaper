---
title: 2025-10-05未命名文件 
grammar_cjkRuby: true
---



<!DOCTYPE html> <html lang="zh-CN"> <head> <meta charset="UTF-8"> <meta name="viewport" content="width=device-width, initial-scale=1.0">  <style> .content-box { max-width: 1800px; margin: -15px auto 5px auto ; padding: 4px 0px 2px 32px; background-color: #818b981f; /* 内容区域可以设置为白色，突出显示 */ border-radius: 6px; } </style> </head> </html>





# 多标签图像分类：
1. [2023 ICCV] **PatchCT: Aligning Patch Set and Label Set with Conditional Transport
for Multi-Label Image Classification**[[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_PatchCT_Aligning_Patch_Set_and_Label_Set_with_Conditional_Transport_ICCV_2023_paper.pdf)
2. [2023 ICCV] **Cdul: Clip-driven unsupervised learning for multi-label image classification**[[paper]](https://arxiv.org/pdf/2307.16634)[[code]](https://github.com/cs-mshah/CDUL)
3. [2024 ICML] **Language-driven Cross-modal Classifier for
Zero-shot Multi-label Image Recognition**[[paper]](https://openreview.net/pdf?id=sHswzNWUW2)[[code]](https://github.com/yic20/CoMC)
4. [2024 AAAI] **TagCLIP: A Local-to-Global Framework to Enhance Open-Vocabulary Multi-Label Classification of CLIP Without Training**[[paper]](https://arxiv.org/pdf/2312.12828)[[code]](https://github.com/linyq2117/TagCLIP)
5. [2025 CVPR] **SPARC: Score Prompting and Adaptive Fusion for Zero-Shot Multi-Label Recognition in Vision-Language Models**[[paper]](https://arxiv.org/pdf/2502.16911?)[[code]](https://github.com/kjmillerCURIS/SPARC)
6. [2025 CVPR] **Classifier-guided CLIP Distillation for Unsupervised Multi-label Classification**[[paper]](https://arxiv.org/pdf/2503.16873)[[code]](https://github.com/k0u-id/CCD)
7. [2025 CVPR] **Recover and Match: Open-Vocabulary Multi-Label Recognition through
Knowledge-Constrained Optimal Transport**[[paper]](https://arxiv.org/pdf/2503.15337)[[code]](https://github.com/EricTan7/RAM)
8. [2025 CVPR] **Correlative and Discriminative Label Grouping for Multi-Label
Visual Prompt Tuning**[[paper]](https://arxiv.org/pdf/2504.09990)
9. [2025 ICML] **From Local Details to Global Context:Advancing Vision-Language Models with Attention-Based Selection**[[paper]](https://arxiv.org/pdf/2505.13233?)[[code]](https://github.com/BIT-DA/ABS)
10. [2025 WACV] **Pay Attention to Your Neighbours:Training-Free Open-Vocabulary Semantic Segmentation**[[paper]](https://arxiv.org/pdf/2404.08181?)[[code]](https://github.com/sinahmr/NACLIP)
11. [2025 ICCV] **Category-Specific Selective Feature Enhancement for Long-Tailed Multi-Label Image Classification**
13. [2025 ICCV]MambaML: Exploring State Space Models for Multi-Label Image Classification
14. [2025 ICCV]Category-Specific Selective Feature Enhancement for Long-Tailed Multi-Label Image Classification
15. [2025 ICCV] **More Reliable Pseudo-labels, Better Performance: A Generalized Approach to Single Positive Multi-label Learning**[[paper]](https://arxiv.org/pdf/2508.20381)
16. [2025 ICCV]Language-Driven Multi-Label Zero-Shot Learning with Semantic Granularity
17. [2024 ICLR] **A Hard-to-Beat Baseline for Training-free CLIP-based Adaptation**[[paper]](https://arxiv.org/pdf/2402.04087)[[code]](https://github.com/mrflogs/ICLR24)
>无需训练”的CLIP模型自适应新方法
18. [2022 IJCV] **Learning to Prompt for Vision-Language Models**[[paper]](https://arxiv.org/pdf/2109.01134v3)[[code]](https://github.com/KaiyangZhou/CoOp)

# 线性特征对齐：
1. [2023 ICCV] **Black Box Few-Shot Adaptation for Vision-Language models**[[paper]](https://arxiv.org/pdf/2304.01752v3)[[code]](https://github.com/saic-fi/LFA)




# Training-Free Open-Vocabulary Semantic Segmentation
<a name="Training-Free_OVSS"></a>
1. [2024 CVPR] **Clip-diy: Clip dense inference yields open-vocabulary semantic segmentation for-free** [[paper]](https://arxiv.org/pdf/2309.14289)
2. [2024 CVPR] **Training-Free Open-Vocabulary Segmentation with Offline Diffusion-Augmented Prototype Generation** [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10655445&tag=1) [[code]](https://github.com/aimagelab/freeda)
<body><div class="content-box"> <span>离线原型生成阶段:使用 Stable Diffusion 模型,结合大量的文本描述提取局部化掩码,采用 DINOv2视觉原型提取,CLIP 文本键提取,每个文本键都与一个视觉原型相关联,构建一个大规模的文本-视觉原型集合 无训练掩码预测阶段:给定一组文本类别,检索到的原型取平均得到视觉原型.超像素的局部区域分割Felzenszwalb,CLIP的全局相似性加权。</span></div></body>


4. [2024 ECCV] **Diffusion Models for Open-Vocabulary Segmentation** [[paper]](https://arxiv.org/pdf/2306.09316) [[code]](https://github.com/karazijal/ovdiff)
5. [2024 ECCV] **ClearCLIP: Decomposing CLIP Representations for Dense Vision-Language Inference** [[paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06346.pdf) [[code]](https://github.com/mc-lan/ClearCLIP)
<body><div class="content-box"> <span>CLIP产生了具有错误分割区域的嘈杂分割图，去除残差连接、实现自注意力和丢弃前馈网络。ClearCLIP 始终如一地生成更清晰、更准确的分割图 </span></div></body>

5. [2024 ECCV] **SCLIP: Rethinking Self-Attention for Dense Vision-Language Inference** [[paper]](https://arxiv.org/pdf/2312.01597) [[code]](https://github.com/wangf3014/SCLIP)
<body><div class="content-box"> <span>CLIP的分割性能不佳是由斑块表示的空间错位引起的，而不是无法提取密集的视觉特征,问题出在CLIP的自注意力模块,使用 CSA 模块代替 CLIP 视觉编码器中的原始自注意力块</span></div></body>

6. [2024 ECCV] **Pay Attention to Your Neighbours: Training-Free Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2404.08181) [[code]](https://github.com/sinahmr/NACLIP)
7. [2024 ECCV] **Proxyclip: Proxy attention improves clip for open-vocabulary segmentation** [[paper]](https://arxiv.org/pdf/2408.04883) [[code]](https://github.com/mc-lan/ProxyCLIP?tab=readme-ov-file) 
8. [2024 ECCV] **Explore the Potential of CLIP for Training-Free Open Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2407.08268) [[code]](https://github.com/leaves162/CLIPtrase)
9. [2024 arXiv] **CLIPer: Hierarchically Improving Spatial Representation of CLIP for Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2411.13836) [[code]](https://github.com/linsun449/cliper.code?tab=readme-ov-file)
10. [2025 CVPR] **LPOSS: Label Propagation Over Patches and Pixels for Open-vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2503.19777) [[code]](https://github.com/vladan-stojnic/LPOSS)
11. [2025 CVPR] **ResCLIP: Residual Attention for Training-free Dense Vision-language Inference** [[paper]](https://arxiv.org/pdf/2411.15851) [[code]](https://github.com/yvhangyang/ResCLIP?tab=readme-ov-file)
<body><div class="content-box"> <span>残差互相关自注意力 （RCS） 和语义反馈细化 （SFR） 模块。这两个模块可以纠正最后一层的注意力，以捕获特定类的特征和局部一致性，从而改进密集视觉语言预测任务的CLIP模型。</span></div></body>

12. [2025 CVPR] **Distilling Spectral Graph for Object-Context Aware Open-Vocabulary Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Kim_Distilling_Spectral_Graph_for_Object-Context_Aware_Open-Vocabulary_Semantic_Segmentation_CVPR_2025_paper.pdf) [[code]](https://github.com/MICV-yonsei/CASS)
13. [2025 CVPR] **Cheb-GR: Rethinking k-nearest neighbor search in Re-ranking for Person Re-identification** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_Cheb-GR_Rethinking_K-nearest_Neighbor_Search_in_Re-ranking_for_Person_Re-identification_CVPR_2025_paper.pdf) [[code]](https://github.com/Jinxi-Yang-WHU/Fast-GCR.git) [[note]](本文提到的很多re-ranking的技术就是对直接计算的相似度矩阵进行更新，前面公式搞了一大堆，最后就是一个特征传播。)
14. [2025 CVPR] **ITACLIP: Boosting Training-Free Semantic Segmentation with Image, Text, and Architectural Enhancements** [[paper]](https://openaccess.thecvf.com/content/CVPR2025W/PixFoundation/papers/Aydin_ITACLIP_Boosting_Training-Free_Semantic_Segmentation_with_Image_Text_and_Architectural_CVPRW_2025_paper.pdf) [[code]](https://github.com/m-arda-aydn/ITACLIP)
15. [2025 CVPR] **Search and Detect: Training-Free Long Tail Object Detection via Web-Image Retrieval** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Sidhu_Search_and_Detect_Training-Free_Long_Tail_Object_Detection_via_Web-Image_CVPR_2025_paper.pdf) [[code]](https://github.com/Mankeerat/SearchDet)
16. [2025 CVPR] **Shift the Lens: Environment-Aware Unsupervised Camouflaged Object Detection** [[paper]]([https://github.com/xiaohainku/EASE](https://openaccess.thecvf.com/content/CVPR2025/papers/Du_Shift_the_Lens_Environment-Aware_Unsupervised_Camouflaged_Object_Detection_CVPR_2025_paper.pdf)) [[code]](https://github.com/xiaohainku/EASE)
17. [2025 ICCV] **LUDVIG: Learning-free Uplifting of 2D Visual features to Gaussian Splatting scene** [[paper]](https://arxiv.org/pdf/2410.14462#page=17.85) [[code]](https://github.com/naver/ludvig)
18. [2025 ICCV] **WildSeg3D: Segment Any 3D Objects in the Wild from 2D Images** [[paper]](https://arxiv.org/pdf/2503.08407)
19. [2025 ICCV] **Harnessing Vision Foundation Models for High-Performance, Training-Free Open Vocabulary Segmentation** [[paper]](https://arxiv.org/pdf/2411.09219) [[code]](https://github.com/YuHengsss/Trident)
20. [2025 ICCV] **E-SAM: Training-Free Segment Every Entity Model** [[paper]](https://arxiv.org/pdf/2503.12094)
21. [2025 ICCV] **ReME: A Data-Centric Framework for Training-Free Open-Vocabulary Segmentation** [[paper]](https://arxiv.org/pdf/2506.21233) [[code]](https://github.com/xiweix/ReME)
<body><div class="content-box"> <span>其核心在于从真实图像中构建一个高质量、对齐良好、语义丰富且上下文相关的片段-文本对参考集，并通过简单的相似度检索机制，在不进行任何模型微调的情况下，显著提升 OVS 性能。</span></div></body>

22. [2025 ICCV] **CorrCLIP: Reconstructing Patch Correlations in CLIP for Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2411.10086) [[code]](https://github.com/zdk258/CorrCLIP)
23. [2025 ICCV] **CCL-LGS: Contrastive Codebook Learning for 3D Language Gaussian Splatting** [[paper]](https://arxiv.org/pdf/2505.20469) [[code]](https://epsilontl.github.io/CCL-LGS/)
24. [2025 ICCV] **Auto-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2312.04539)
25. [2025 ICCV] **Understanding Personal Concept in Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2507.11030)
26. [2025 ICCV] **Training-Free Class Purification for Open-Vocabulary Semantic Segmentation**
<body><div class="content-box"> <span>FreeCP通过分析精炼前后类激活图的空间一致性，分两阶段进行类别净化：首先滤除冗余类别，然后结合大型语言模型（LLM）生成的细粒度描述来消除视觉-语言歧义。即插即用</span></div></body>

27. [2025 ICCV] **DIH-CLIP: Unleashing the Diversity of Multi-Head Self-Attention for Training-Free Open-Vocabulary Semantic Segmentation**
28. [2025 ICCV] **Correspondence as Video: Test-Time Adaption on SAM2 for Reference Segmentation in the Wild**
29. [2025 ICCV] **Feature Purification Matters: Suppressing Outlier Propagation for Training-Free Open-Vocabulary Semantic Segmentation**
30 [2025 ICCV] **Plug-in Feedback Self-adaptive Attention in CLIP for Training-free Open-Vocabulary Segmentation**
31. [2025 ICCV] **Test-Time Retrieval-Augmented Adaptation for Vision-Language Models** [[paper]](https://openreview.net/pdf?id=V3zobHnS61) [[code]](https://github.com/xinqi-fan/TT-RAA)
32. [2025 ICCV] **Text-guided Visual Prompt DINO for Generic Segmentation**
33. [2025 ICCV] **SCORE: Scene Context Matters in Open-Vocabulary Remote Sensing Instance Segmentation** [[paper]](https://arxiv.org/pdf/2507.12857) [[code]](https://github.com/HuangShiqi128/SCORE)
34. [2025 ICCV] **Images as Noisy Labels: Unleashing the Potential of the Diffusion Model for Open-Vocabulary Semantic Segmentation**
35. [2025 AAAI] **Training-free Open-Vocabulary Semantic Segmentation via Diverse Prototype Construction and Sub-region Matching** [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/33137)
36. [2025 arXiv] **Self-Calibrated CLIP for Training-Free Open-Vocabulary Segmentation** [[paper]](https://arxiv.org/pdf/2411.15869) [[code]](https://github.com/SuleBai/SC-CLIP?tab=readme-ov-file)
37. [2025 arXiv] **Test-Time Adaptation of Vision-Language Models for Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2505.21844v1) [[code]](https://github.com/dosowiechi/MLMP?tab=readme-ov-file)
38. [2025 arXiv] **FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/abs/2504.10487) [[code]](https://github.com/yasserben/FLOSS)
39. [2025 arXiv] **TextRegion: Text-Aligned Region Tokens from Frozen Image-Text Models** [[paper]](https://arxiv.org/pdf/2505.23769) [[code]](https://github.com/avaxiao/TextRegion)
<body><div class="content-box"> <span>SAM2与图像-文本模型(CLIP SigLIP2)结合</span></div></body>


41. [2025 arXiv] **A Survey on Training-free Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2505.22209)
42. [2025 arXiv] **No time to train! Training-Free Reference-Based Instance Segmentation** [[paper]](https://arxiv.org/pdf/2507.02798) [[code]](https://github.com/miquel-espinosa/no-time-to-train?tab=readme-ov-file) [[note]](https://mp.weixin.qq.com/s/6BjzduZoqc2OgocNkaDiPQ)
<body><div class="content-box"> <span>DINOv2与SAM2结合少样本，使用DINOv2提取特征，求类平均值特征。</span></div></body>


# Training Open-Vocabulary Semantic Segmentation
<a name="Training_OVSS"></a>
1. [2022 CVPR] **GroupViT: Semantic Segmentation Emerges from Text Supervision** [[paper]](https://arxiv.org/pdf/2202.11094) [[code]](https://github.com/NVlabs/GroupViT?tab=readme-ov-file)
2. [2023 CVPR] **Open Vocabulary Semantic Segmentation with Patch Aligned Contrastive Learning** [[paper]](https://arxiv.org/pdf/2212.04994)
3. [2023 CVPR] **Learning to Generate Text-grounded Mask for Open-world Semantic Segmentation from Only Image-Text Pairs** [[paper]](https://arxiv.org/pdf/2212.00785) [[code]](https://github.com/khanrc/tcl?tab=readme-ov-file)
4. [2023 ICCV] **Exploring Open-Vocabulary Semantic Segmentation from CLIP Vision Encoder Distillation Only** [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Exploring_Open-Vocabulary_Semantic_Segmentation_from_CLIP_Vision_Encoder_Distillation_Only_ICCV_2023_paper.pdf)
5. [2023 ICML] **SegCLIP: Patch Aggregation with Learnable Centers for Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2211.14813) [[code]](https://github.com/ArrowLuo/SegCLIP?tab=readme-ov-file)
6. [2023 ICML] **Grounding Everything: Emerging Localization Properties in Vision-Language Transformers** [[paper]](https://arxiv.org/pdf/2312.00878) [[code]](https://github.com/WalBouss/GEM?tab=readme-ov-file)
7. [2023 NIPS] **Uncovering Prototypical Knowledge for Weakly Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2310.19001) [[code]](https://github.com/Ferenas/PGSeg?tab=readme-ov-file)
8. [2024 CVPR] **SED: A Simple Encoder-Decoder for Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2311.15537v2) [[code]](https://github.com/xb534/SED.git)
9. [2024 CVPR] **Not All Classes Stand on Same Embeddings: Calibrating a Semantic Distance with Metric Tensor** [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10658112&tag=1)
10. [2024 CVPR] **USE: Universal Segment Embeddings for Open-Vocabulary Image Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_USE_Universal_Segment_Embeddings_for_Open-Vocabulary_Image_Segmentation_CVPR_2024_paper.pdf)
11. [2024 CVPR] **CAT-Seg: Cost Aggregation for Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2303.11797) [[code]](https://github.com/cvlab-kaist/CAT-Seg)
12. [2024 CVPR] **Emergent Open-Vocabulary Semantic Segmentation from Off-the-shelf Vision-Language Models** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Luo_Emergent_Open-Vocabulary_Semantic_Segmentation_from_Off-the-shelf_Vision-Language_Models_CVPR_2024_paper.pdf) [[code]](https://github.com/letitiabanana/PnP-OVSS)
13. [2024 CVPR] **SAM-CLIP: Merging Vision Foundation Models towards Semantic and Spatial Understanding** [[paper]](https://openaccess.thecvf.com/content/CVPR2024W/ELVM/papers/Wang_SAM-CLIP_Merging_Vision_Foundation_Models_Towards_Semantic_and_Spatial_Understanding_CVPRW_2024_paper.pdf)
14. [2024 CVPR] **Image-to-Image Matching via Foundation Models: A New Perspective for Open-Vocabulary Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Image-to-Image_Matching_via_Foundation_Models_A_New_Perspective_for_Open-Vocabulary_CVPR_2024_paper.pdf)
16. [2024 ECCV] **CLIP-DINOiser: Teaching CLIP a few DINO tricks for open-vocabulary semantic segmentation** [[paper]](https://arxiv.org/pdf/2312.12359) [[code]](https://github.com/wysoczanska/clip_dinoiser)
17. [2024 ECCV] **In Defense of Lazy Visual Grounding for Open-Vocabulary Semantic Segmentation** [[paper]](http://arxiv.org/abs/2408.04961) [[code]](https://github.com/dahyun-kang/lavg)
18. [2024 ICLR] **CLIPSelf: Vision Transformer Distills Itself for Open-Vocabulary Dense Prediction** [[paper]](https://arxiv.org/pdf/2310.01403v2) [[code]](https://github.com/wusize/CLIPSelf?tab=readme-ov-file)
19. [2024 NIPS] **Towards Open-Vocabulary Semantic Segmentation Without Semantic Labels** [[paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/1119587863e78451f080da2a768c4935-Paper-Conference.pdf) [[code]](https://github.com/cvlab-kaist/PixelCLIP)
20. [2024 NIPS] **Relationship Prompt Learning is Enough for Open-Vocabulary Semantic Segmentation** [[paper]](https://openreview.net/pdf?id=PKcCHncbzg)
21. [2024 arXiv] **DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment** [[paper]](https://arxiv.org/pdf/2412.16334)
22. [2025 CVPR] **Semantic Library Adaptation: LoRA Retrieval and Fusion for Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2503.21780)
23. [2025 CVPR] **Your ViT is Secretly an Image Segmentation Model** [[paper]](https://arxiv.org/pdf/2503.19108) [[code]](https://github.com/tue-mps/eomt)
24. [2025 CVPR] **Exploring Simple Open-Vocabulary Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Lai_Exploring_Simple_Open-Vocabulary_Semantic_Segmentation_CVPR_2025_paper.pdf)
25. [2025 CVPR] **Dual Semantic Guidance for Open Vocabulary Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Dual_Semantic_Guidance_for_Open_Vocabulary_Semantic_Segmentation_CVPR_2025_paper.pdf)
26. [2025 CVPR] **Multi-Label Prototype Visual Spatial Search for Weakly Supervised Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Duan_Multi-Label_Prototype_Visual_Spatial_Search_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2025_paper.pdf)
27. [2025 CVPR] **DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_DeCLIP_Decoupled_Learning_for_Open-Vocabulary_Dense_Perception_CVPR_2025_paper.pdf) [[code]](https://github.com/xiaomoguhz/DeCLIP)

<body><div class="content-box"> <span>DeCLIP 通过解耦自注意力模块并分别对内容特征和上下文特征进行蒸馏，有效地解决了 CLIP 在开放词汇密集感知任务中局部特征辨别力不足和空间一致性差的问题。</span></div></body>

29. [2025 ICCV] **Talking to DINO: Bridging Self-Supervised Vision Backbones with Language for Open-Vocabulary Segmentation** [[paper]](https://arxiv.org/pdf/2411.19331) [[code]](https://github.com/lorebianchi98/Talk2DINO)
<body><div class="content-box"> <span>映射函数，CLIP文本映射到DINOv2特征向量，训练矩阵，DINOv2特征向量直接映射到CLIP提取的文本特征</span></div></body>


31. [2025 ICLR] **Cross the Gap: Exposing the Intra-modal Misalignment in CLIP via Modality Inversion** [[paper]](https://arxiv.org/pdf/2502.04263) [[code]](https://github.com/miccunifi/Cross-the-Gap?tab=readme-ov-file)
# Zero-Shot Open-Vocabulary Semantic Segmentation
<a name="ZSOVSS"></a>
1. [2024 CVPR] **On the test-time zero-shot generalization of vision-language models: Do we really need prompt learning?** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Zanella_On_the_Test-Time_Zero-Shot_Generalization_of_Vision-Language_Models_Do_We_CVPR_2024_paper.pdf) [[code]](https://github.com/MaxZanella/MTA)
2. [2024 CVPR] **Exploring Regional Clues in CLIP for Zero-Shot Semantic Segmentation** [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10656627&tag=1) [[code]](https://github.com/Jittor/JSeg)
3. [2024 CVPR] **Diffuse, Attend, and Segment: Unsupervised Zero-Shot Segmentation using Stable Diffusion** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Tian_Diffuse_Attend_and_Segment_Unsupervised_Zero-Shot_Segmentation_using_Stable_Diffusion_CVPR_2024_paper.pdf) [[code]](https://github.com/google/diffseg)
4. [2024 ECCV] **OTSeg: Multi-prompt Sinkhorn Attention for Zero-Shot Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2403.14183) [[code]](https://github.com/cubeyoung/OTSeg?tab=readme-ov-file)
5. [2024 ICCV] **Zero-guidance Segmentation Using Zero Segment Labels** [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Rewatbowornwong_Zero-guidance_Segmentation_Using_Zero_Segment_Labels_ICCV_2023_paper.pdf) [[code]](https://github.com/nessessence/ZeroGuidanceSeg)

<body><div class="content-box"> <span>DINO-ViT 模型提取图像的深层像素级特征，聚类得到分割掩码，输入图像和通过聚类得到的二值掩码会被同时送入CLIP，新颖的注意力掩码（Attention Masking）技术，特别是全局消减（Global Subtraction），使用 ZeroCap 把图像生成文本，相似度分数是视觉嵌入和预测文本嵌入（通过 CLIP 文本编码器计算）的余弦相似度的平均值</span></div></body>

7. [2024 NIPS] **DiffCut: Catalyzing Zero-Shot Semantic Segmentation with Diffusion Features and Recursive Normalized Cut** [[paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/1867748a011e1425b924ec72a4066b62-Paper-Conference.pdf) [[code]](https://github.com/PaulCouairon/DiffCut)
8. [2025 ICLR] **Efficient and Context-Aware Label Propagation for Zero-/Few-Shot Training-Free Adaptation of Vision-Language Model** [[paper]](https://arxiv.org/pdf/2412.18303) [[code]](https://github.com/Yushu-Li/ECALP?tab=readme-ov-file)
# Few-Shot Open-Vocabulary Semantic Segmentation
<a name="FSOVSS"></a>
1. [2024 NIPS] **Training-Free Open-Ended Object Detection and Segmentation via Attention as Prompts** [[paper]](https://arxiv.org/pdf/2410.05963)
2. [2024 NIPS] **A Surprisingly Simple Approach to Generalized Few-Shot Semantic Segmentation** [[paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/2f75a57e9c71e8369da0150ea769d5a2-Paper-Conference.pdf) [[code]](https://github.com/IBM/BCM)
3. [2024 NIPS] **Renovating Names in Open-Vocabulary Segmentation Benchmarks** [[paper]](https://openreview.net/pdf?id=Uw2eJOI822) [[code]](https://andrehuang.github.io/renovate/)
4. [2025 CVPR] **Hyperbolic Uncertainty-Aware Few-Shot Incremental Point Cloud Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Sur_Hyperbolic_Uncertainty-Aware_Few-Shot_Incremental_Point_Cloud_Segmentation_CVPR_2025_paper.pdf)
5. [2025 ICCV] **Probabilistic Prototype Calibration of Vision-language Models for Generalized Few-shot Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2506.22979) [[code]](https://github.com/jliu4ai/FewCLIP)
6. [2025 MICCAI] **Realistic Adaptation of Medical Vision-Language Models** [[paper]](https://arxiv.org/pdf/2506.17500) [[code]](https://github.com/jusiro/SS-Text)

# Supervised Semantic Segmentation
<a name="Supervised_Semantic_Segmentation"></a>
1. [2021 ICCV] **Vision Transformers for Dense Prediction** [[paper]](https://arxiv.org/abs/2103.13413v1) [[code]](https://github.com/isl-org/DPT?tab=readme-ov-file)
2. [2021 ICCV] **Segmenter: Transformer for Semantic Segmentation** [[paper]](https://arxiv.org/abs/2105.05633) [[code]](https://github.com/rstrudel/segmenter)
3. [2022 ICLR] **Language-driven Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2201.03546) [[code]](https://github.com/isl-org/lang-seg)
4. [2025 CVPV] **Your ViT is Secretly an Image Segmentation Model** [[paper]](https://arxiv.org/pdf/2503.19108) [[code]](https://github.com/tue-mps/eomt)
# Weakly Supervised Semantic Segmentation
<a name="Weakly_Supervised_Semantic_Segmentation"></a>
1. [2022 CVPR] **Learning Affinity from Attention: End-to-End Weakly-Supervised Semantic Segmentation with Transformers** [[paper]](https://arXiv.org/pdf/2203.02664) [[code]](https://github.com/rulixiang/afa)
2. [2022 CVPR] **MCTFormer:Multi-Class Token Transformer for Weakly Supervised Semantic Segmentation** [[paper]](https://arXiv.org/pdf/2203.02891) [[code]](https://github.com/xulianuwa/MCTformer)
3. [2023 CVPR] **Learning Multi-Modal Class-Specific Tokens for Weakly Supervised Dense Object Localization** [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Learning_Multi-Modal_Class-Specific_Tokens_for_Weakly_Supervised_Dense_Object_Localization_CVPR_2023_paper.pdf) [[code]](https://github.com/xulianuwa/MMCST)
4. [2023 ICCV] **Spatial-Aware Token for Weakly Supervised Object Localization** [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_Spatial-Aware_Token_for_Weakly_Supervised_Object_Localization_ICCV_2023_paper.pdf) [[code]](https://github.com/wpy1999/SAT)
5. [2023 CVPR] **Boundary-enhanced Co-training for Weakly Supervised Semantic Segmentatio** [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Rong_Boundary-Enhanced_Co-Training_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2023_paper.pdf) [[code]](https://github.com/ShenghaiRong/BECO?tab=readme-ov-file)
6. [2023 CVPR] **ToCo:Token Contrast for Weakly-Supervised Semantic Segmentation** [[paper]](https://arXiv.org/pdf/2303.01267) [[code]](https://github.com/rulixiang/ToCo)
7. [2023 CVPR] **CLIP is Also an Efficient Segmenter: A Text-Driven Approach for Weakly Supervised Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2212.09506) [[code]](https://github.com/linyq2117/CLIP-ES)
8. [2023 arXiv] **MCTformer+: Multi-Class Token Transformer for Weakly Supervised Semantic Segmentation** [[paper]](https://arXiv.org/pdf/2308.03005) [[code]](https://github.com/xulianuwa/MCTformer)
9. [2024 CVPR] **Frozen CLIP: A Strong Backbone for Weakly Supervised Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2406.11189v1) [[code]](https://github.com/zbf1991/WeCLIP)
10. [2024 CVPR] **CorrMatch: Label Propagation via Correlation Matching for Semi-Supervised Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2411.13147v1) [[code]](https://github.com/ZiqinZhou66/ZegCLIP?tab=readme-ov-file)
11. [2024 CVPR] **DuPL: Dual Student with Trustworthy Progressive Learning for Robust Weakly Supervised Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2403.11184) [[code]](https://github.com/Wu0409/DuPL)
12. [2024 CVPR] **Hunting Attributes: Context Prototype-Aware Learning for Weakly Supervised Semantic Segmentation** [[paper]](https:https://openaccess.thecvf.com/content/CVPR2024/papers/Tang_Hunting_Attributes_Context_Prototype-Aware_Learning_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2024_paper.pdf) [[code]](https://github.com/Barrett-python/CPAL)
13. [2024 CVPR] **Improving the Generalization of Segmentation Foundation Model under Distribution Shift via Weakly Supervised Adaptation** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Improving_the_Generalization_of_Segmentation_Foundation_Model_under_Distribution_Shift_CVPR_2024_paper.pdf)
14. [2024 CVPR] **Official code for Class Tokens Infusion for Weakly Supervised Semantic Segmentation** [[paper]](Yoon_Class_Tokens_Infusion_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2024_paper) [[code]](https://github.com/yoon307/CTI)
15. [2024 CVPR] **Separate and Conquer: Decoupling Co-occurrence via Decomposition and Representation for Weakly Supervised Semantic Segmentation** [[paper]](https://arXiv.org/pdf/2402.18467) [[code]](https://github.com/zwyang6/SeCo)
16. [2024 CVPR] **Class Tokens Infusion for Weakly Supervised Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Yoon_Class_Tokens_Infusion_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2024_paper.pdf) [[code]](https://github.com/yoon307/CTI)
17. [2024 CVPR] **SFC: Shared Feature Calibration in Weakly Supervised Semantic Segmentation** [[paper]](https://arXiv.org/pdf/2401.11719) [[code]](https://github.com/Barrett-python/SFC)
18. [2024 CVPR] **PSDPM:Prototype-based Secondary Discriminative Pixels Mining for Weakly Supervised Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_PSDPM_Prototype-based_Secondary_Discriminative_Pixels_Mining_for_Weakly_Supervised_Semantic_CVPR_2024_paper.pdf) [[code]](https://github.com/xinqiaozhao/PSDPM)
19. [2024 ECCV] **DIAL: Dense Image-text ALignment for Weakly Supervised Semantic Segmentation** [[paper]](https://arXiv.org/pdf/2409.15801)
20. [2024 ECCV] **CoSa:Weakly Supervised Co-training with Swapping Assignments for Semantic Segmentation** [[paper]](https://arXiv.org/pdf/2402.17891) [[code]](https://github.com/youshyee/CoSA)
21. [2024 AAAI] **Progressive Feature Self-Reinforcement for Weakly Supervised Semantic Segmentation** [[paper]](https://arXiv.org/pdf/2312.08916) [[code]](https://github.com/Jessie459/feature-self-reinforcement)
22. [2024 arXiv] **A Realistic Protocol for Evaluation of Weakly Supervised Object Localization** [[paper]](https://arXiv.org/pdf/2404.10034) [[code]](https://github.com/shakeebmurtaza/wsol_model_selection)
23. [2024 IEEE] **SSC:Spatial Structure Constraints for Weakly Supervised Semantic Segmentation** [[paper]](https://arXiv.org/pdf/2401.11122) [[code]](https://github.com/NUST-Machine-Intelligence-Laboratory/SSC)
24. [2025 CVPR] **POT: Prototypical Optimal Transport for Weakly Supervised Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_POT_Prototypical_Optimal_Transport_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2025_paper.pdf) [[code]](https://github.com/jianwang91/POT)
25. [2025 CVPR] **PROMPT-CAM: A Simpler Interpretable Transformer for Fine-Grained Analysis** [[paper]](https://arXiv.org/pdf/2501.09333) [[code]](https://github.com/Imageomics/Prompt_CAM)
26. [2025 CVPR] **Exploring CLIP’s Dense Knowledge for Weakly Supervised Semantic Segmentation** [[paper]](https://arXiv.org/pdf/2503.20826) [[code]](https://github.com/zwyang6/ExCEL)
27. [2025 CVPR] **GET: Unlocking the Multi-modal Potential of CLIP for Generalized Category Discovery** [[paper]](https://arXiv.org/abs/2403.09974) [[code]](https://github.com/enguangW/GET)
28. [2025 CVPR] **Multi-Label Prototype Visual Spatial Search for Weakly Supervised Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Duan_Multi-Label_Prototype_Visual_Spatial_Search_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2025_paper.pdf)
29. [2025 CVPR] **Prompt Categories Cluster for Weakly Supervised Semantic Segmentation** [[paper]](https://arXiv.org/pdf/2412.13823)
31. [2025 ICCV] **Class Token as Proxy: Optimal Transport-assisted Proxy Learning for Weakly Supervised Semantic Segmentation**
32. [2025 ICCV] **Know Your Attention Maps: Class-specific Token Masking for Weakly Supervised Semantic Segmentation**
33. [2025 ICCV] **Bias-Resilient Weakly Supervised Semantic Segmentation Using Normalizing Flows**
34. [2025 ICCV] **OVA-Fields: Weakly Supervised Open-Vocabulary Affordance Fields for Robot Operational Part Detection**
35. [2025 AAAI] **MoRe: Class Patch Attention Needs Regularization for Weakly Supervised Semantic Segmentation** [[paper]](https://arXiv.org/pdf/2412.11076) [[code]](https://github.com/zwyang6/MoRe)
36. [2025 arXiv] **TeD-Loc: Text Distillation for Weakly Supervised Object Localization** [[paper]](https://arXiv.org/pdf/2501.12632) [[code]](https://github.com/shakeebmurtaza/TeDLOC)
37. [2025 arXiv] **Image Augmentation Agent for Weakly Supervised Semantic Segmentation** [[paper]](https://arXiv.org/pdf/2412.20439)
# Semi-Supervised Semantic Segmentation
<a name="Semi-Supervised_Semantic_Segmentation"></a>
1. [2025 ICCV] **ConformalSAM: Unlocking the Potential of Foundational Segmentation Models in Semi-Supervised Semantic Segmentation with Conformal Prediction** [[paper]](https://arxiv.org/pdf/2507.15803) [[code]](https://github.com/xinqi-fan/TT-RAA)
# Unsupervised Semantic Segmentation
<a name="Unsupervised_Semantic_Segmentation"></a>
1. [2021 ICCV] **Emerging Properties in Self-Supervised Vision Transformers** [[paper]](http://openaccess.thecvf.com//content/ICCV2021/papers/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf) [[code]](https://github.com/facebookresearch/dino) [[note]](https://blog.csdn.net/YoooooL_/article/details/129234966)
2. [2022 CVPR] **Deep Spectral Methods: A Surprisingly Strong Baseline for Unsupervised Semantic Segmentation and Localization** [[paper]](https://arxiv.org/abs/2205.07839) [[code]](https://github.com/lukemelas/deep-spectral-segmentation?tab=readme-ov-file)
3. [2022 CVPR] **Freesolo: Learning to segment objects without annotations** [[paper]](https://arxiv.org/pdf/2202.12181) [[code]](https://github.com/NVlabs/FreeSOLO)
4. [2022 ECCV] **Extract Free Dense Labels from CLIP** [[paper]](https://arxiv.org/pdf/2112.01071) [[code]](https://github.com/chongzhou96/MaskCLIP?tab=readme-ov-file) [[note]](https://www.cnblogs.com/lipoicyclic/p/16967704.html)
5. [2023 CVPR] **ZegCLIP: Towards Adapting CLIP for Zero-shot Semantic Segmentation** [[paper]](https://arxiv.org/abs/2212.03588) [[code]](https://github.com/ZiqinZhou66/ZegCLIP?tab=readme-ov-file)
6. [2024 CVPR] **Guided Slot Attention for Unsupervised Video Object Segmentation** [[paper]](https://arxiv.org/pdf/2303.08314v3) [[code]](https://github.com/Hydragon516/GSANet)
7. [2024 CVPR] **ReCLIP++:Learn to Rectify the Bias of CLIP for Unsupervised Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2408.06747) [[code]](https://github.com/dogehhh/ReCLIP)
8. [2024 CVPR] **CuVLER: Enhanced Unsupervised Object Discoveries through Exhaustive Self-Supervised Transformers** [[paper]](https://arxiv.org/pdf/2403.07700) [[code]](https://github.com/shahaf-arica/CuVLER?tab=readme-ov-file)
9. [2024 CVPR] **EAGLE: Eigen Aggregation Learning for Object-Centric Unsupervised Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2403.01482) [[code]](https://github.com/MICV-yonsei/EAGLE?tab=readme-ov-file)
10. [2024 ECCV] **Unsupervised Dense Prediction using Differentiable Normalized Cuts** [[paper]](https://fq.pkwyx.com/default/https/www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05675.pdf)
11. [2024 NIPS] **PaintSeg: Training-free Segmentation via Painting** [[paper]](https://arxiv.org/abs/2305.19406)
12. [2025 ICCV] **DIP: Unsupervised Dense In-Context Post-training of Visual Representations** [[paper]](https://arxiv.org/pdf/2506.18463) [[code]](https://github.com/sirkosophia/DIP)




# 检索:
1. [2024 ICML] **Cluster-Aware Similarity Diffusion for Instance Retrieval**[[paper]](https://arxiv.org/pdf/2406.02343)
2. [2025 CVPR] **Cheb-GR: Rethinking k-nearest neighbor search in Re-ranking for Person
 Re-identification**[[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_Cheb-GR_Rethinking_K-nearest_Neighbor_Search_in_Re-ranking_for_Person_Re-identification_CVPR_2025_paper.pdf)
3. [2025 NEIGHBOR] **Neighbor-aware Geodesic Transportation for Neighborhood Refinery**[[paper]](https://openreview.net/pdf?id=DWI1xx2sX5)
4. [2021 NIPS] **Contextual Similarity Aggregation with Self-attention for Visual Re-ranking**[[paper]](https://arxiv.org/pdf/2110.13430)
5. [2027 AAAI] **Regularized diffusion process for visual retrieval**
6. [2025 arXiv] **Global-to-Local or Local-to-Global? Enhancing Image Retrieval with Efficient Local Search and Effective Global Re-ranking**[[paper]](https://arxiv.org/pdf/2509.04351)
# 老师提供暂存：
1. [2025 arXiv] **POT: Prototypical Optimal Transport for Weakly Supervised Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_POT_Prototypical_Optimal_Transport_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2025_paper.pdf) 
2. [2025CVPR] **Multi-Label Prototype Visual Spatial Search for Weakly Supervised Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Duan_Multi-Label_Prototype_Visual_Spatial_Search_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2025_paper.pdf)
3. [2024 arXiv]**Toward Modality Gap: Vision Prototype Learning for Weakly-supervised Semantic Segmentation with CLIP** [[paper]](https://arxiv.org/pdf/2412.19650)
5. [2025 arXiv]  **2025-NIPS-Disentangling CLIP for Multi-Object Perception** [[paper]](https://arxiv.org/html/2502.02977v3)
6. [2021 ICLR] **A Trainable Optimal Transport Embedding for Feature Aggregation and its Relationship to Attention**[[paper]](https://arxiv.org/pdf/2006.12065)
7.[2025 ICCV] **Interpretable point cloud classification using multiple instance learning**[[paper]]()














----------

