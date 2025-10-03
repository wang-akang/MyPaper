### 多标签图像分类：
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
12. [unknown] **Self-Calibrated CLIP for Training-Free Open-Vocabulary Segmentation**[[paper]](https://arxiv.org/pdf/2411.15869)[[code]](https://github.com/SuleBai/SC-CLIP)
13. [2025 ICCV]MambaML: Exploring State Space Models for Multi-Label Image Classification
14. [2025 ICCV]Category-Specific Selective Feature Enhancement for Long-Tailed Multi-Label Image Classification
15. [2025 ICCV] **More Reliable Pseudo-labels, Better Performance: A Generalized Approach to Single Positive Multi-label Learning**[[paper]](https://arxiv.org/pdf/2508.20381)
16. [2025 ICCV]Language-Driven Multi-Label Zero-Shot Learning with Semantic Granularity
17. [2024 ICLR] **A Hard-to-Beat Baseline for Training-free CLIP-based Adaptation**[[paper]](https://arxiv.org/pdf/2402.04087)[[code]](https://github.com/mrflogs/ICLR24)
>无需训练”的CLIP模型自适应新方法
18. [2022 IJCV] **Learning to Prompt for Vision-Language Models**[[paper]](https://arxiv.org/pdf/2109.01134v3)[[code]](https://github.com/KaiyangZhou/CoOp)
19. [2025 CVPR] **DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception**[[paper]](https://arxiv.org/pdf/2505.04410)[[code]](https://github.com/xiaomoguhz/DeCLIP)
>DeCLIP 通过解耦自注意力模块并分别对内容特征和上下文特征进行蒸馏，有效地解决了 CLIP 在开放词汇密集感知任务中局部特征辨别力不足和空间一致性差的问题。
### 线性特征对齐：
1. [2025 ICCV]**Black Box Few-Shot Adaptation for Vision-Language models**[[paper]](https://arxiv.org/pdf/2304.01752v3)[[code]](https://github.com/saic-fi/LFA)

### 语义分割:
1. [2025 CVPR] **Test-Time Adaptation of Vision-Language Models forOpen-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2505.21844v1)[[code]](https://github.com/dosowiechi/MLMP)
2. [2025 ICCV] **Optimal Transport-assisted Proxy Learning for Weakly Supervised Semantic Segmentation**[[paper]](https://iccv.thecvf.com/virtual/2025/poster/1933)
3. [2025 ICCV] **Know Your Attention Maps: Class-specific Token Masking for Weakly Supervised Semantic Segmentation**[[paper]](https://arxiv.org/html/2507.06848v1)
4. [2025 ICCV] **Images as Noisy Labels: Unleashing the Potential of the Diffusion Model for Open-Vocabulary Semantic Segmentation**[[paper]](https://iccv.thecvf.com/virtual/2025/poster/645)
5. [2025 NIPS] **Disentangling CLIP for Multi-Object Perception**[[paper]](https://arxiv.org/pdf/2502.02977v3)
>VLMs 中类别特征之间的高 MFI 严重影响了其多对象感知能力。通过引入 MFI Loss(解耦文本特征) 和 ASL Loss 来解耦 CLIP 特征训练目标降低总训练损失是 MFI 损失和 ASL 损失的组合
6. [2023 ICCV] **Zero-guidance Segmentation Using Zero Segment Labels**[[paper]](https://arxiv.org/pdf/2303.13396)
>DINO-ViT 模型提取图像的深层像素级特征，聚类得到分割掩码，输入图像和通过聚类得到的二值掩码会被同时送入CLIP，新颖的注意力掩码（Attention Masking）技术，特别是全局消减（Global Subtraction），使用 ZeroCap 把图像生成文本，相似度分数是视觉嵌入和预测文本嵌入（通过 CLIP 文本编码器计算）的余弦相似度的平均值
7. [2025 ICCV] **Enhancing Few-Shot Vision-Language Classification with Large MultimodalModel Features**[[paper]](https://arxiv.org/pdf/2412.00142)
8. [2025 CVPR] **Generalized Few-shot 3D Point Cloud Segmentation with Vision-Language Model**[[paper]](https://arxiv.org/pdf/2503.16282)
9. [2025 ICCV] **DictAS: A Framework for Class-Generalizable Few-Shot Anomaly Segmentation via Dictionary**[[paper]](https://www.arxiv.org/pdf/2508.13560)[[code]](https://github.com/xiaozhen228/DictAS)
10. [2025 arXiv] **No time to train! Training-Free Reference-Based Instance Segmentation**[[paper]](https://arxiv.org/pdf/2507.02798)[[code]](https://github.com/miquel-espinosa/no-time-to-train) 👌
>DINOv2与SAM2结合少样本分割
12. [2025 ICCV] **DenseVLM: A Retrieval and Decoupled Alignment Framework for Open-Vocabulary Dense Prediction**[[paper]](https://arxiv.org/pdf/2412.06244)[[code]](https://link.zhihu.com/?target=https%3A//github.com/HVision-NKU/DenseVLM)
>解决前景偏差问题
13. [2025 arXiv] **TextRegion: Text-Aligned Region Tokens from Frozen Image-Text Models**[[paper]](https://arxiv.org/pdf/2505.23769v1)[[code]](https://github.com/avaxiao/TextRegion)
>SAM2与图像-文本模型(CLIP SigLIP2)结合
14. [2025 ICCV] **CorrCLIP: Reconstructing Patch Correlations in CLIP for Open-Vocabulary Semantic Segmentation**[[paper]](https://arxiv.org/pdf/2411.10086)[[code]](https://github.com/zdk258/CorrCLIP)
>CLIP与SAM结合CorrCLIP 利用分段任意模型 （SAM） 来定义补丁交互的范围，从而减少类间相关性
15. [2024 CVPR] **Training-Free Open-Vocabulary Segmentation with Offline Diffusion-Augmented Prototype Generation**[[paper]](https://arxiv.org/pdf/2404.06542)[[code]](https://github.com/aimagelab/freeda/)
> **离线原型生成阶段**:使用 Stable Diffusion 模型,结合大量的文本描述提取局部化掩码,采用 DINOv2视觉原型提取,CLIP 文本键提取,每个文本键都与一个视觉原型相关联,构建一个大规模的文本-视觉原型集合
 **无训练掩码预测阶段**:给定一组文本类别,检索到的原型取平均得到视觉原型.超像素的局部区域分割Felzenszwalb,CLIP的全局相似性加权.
16. [2025 NeurIPS] **SANSA: Unleashing the Hidden Semantics in SAM2for Few-Shot Segmentation**[[paper]](https://arxiv.org/pdf/2505.21795)[[code]](https://github.com/ClaudiaCuttano/SANSA)
17. [2025 arXiv] **X-SAM: From Segment Anything to Any Segmentation**[[paper]](https://arxiv.org/pdf/2508.04655)[[code]](https://github.com/wanghao9610/X-SAM)
18. [2025 CVPR] **Segment Any Motion in Videos**[[paper]](https://arxiv.org/pdf/2503.22268)[[code]](https://github.com/nnanhuang/SegAnyMo)
19. [2025 NeurIPS] **OpenWorldSAM: Extending SAM2 for Universal Image Segmentation with Language Prompts**[[paper]](https://arxiv.org/pdf/2507.05427)
20. [2025 ICCV] **Talking to DINO: Bridging Self-Supervised Vision Backbones with Language for Open-Vocabulary Segmentation**[[paper]](https://arxiv.org/pdf/2411.19331)[[code]](https://lorebianchi98.github.io/Talk2DINO/)
21. [2025 ICIP] **Zero-Shot Pseudo Labels Generation Using SAM and CLIP for Semi-Supervised Semantic Segmentation**[[paper]](https://arxiv.org/pdf/2505.19846)
>使用SAM和CLIP通过zero-shot标注生成伪标签，并通过UniMatch的半监督学习框架作为增强标签来提高其质量。


### 检索:
1. [2024 ICML] **Cluster-Aware Similarity Diffusion for Instance Retrieval**[[paper]](https://arxiv.org/pdf/2406.02343)
2. [2025 CVPR] **Cheb-GR: Rethinking k-nearest neighbor search in Re-ranking for Person
 Re-identification**[[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_Cheb-GR_Rethinking_K-nearest_Neighbor_Search_in_Re-ranking_for_Person_Re-identification_CVPR_2025_paper.pdf)
3. [2025 NEIGHBOR] **Neighbor-aware Geodesic Transportation for Neighborhood Refinery**[[paper]](https://openreview.net/pdf?id=DWI1xx2sX5)
4. [2021 NIPS] **Contextual Similarity Aggregation with Self-attention for Visual Re-ranking**[[paper]](https://arxiv.org/pdf/2110.13430)
5. [2027 AAAI] **Regularized diffusion process for visual retrieval**
6. [2025 arXiv] **Global-to-Local or Local-to-Global? Enhancing Image Retrieval with Efficient Local Search and Effective Global Re-ranking**[[paper]](https://arxiv.org/pdf/2509.04351)




### few-shot：
1. [2025 ICCV] **Object-level Correlation for Few-Shot Segmentation**[[paper]](https://arxiv.org/pdf/2509.07917)
2. [2025 ICCV] **When Pixel Difference Patterns Meet ViT: PiDiViT for Few-Shot Object Detection**
3. [2025 ICCV] **Probabilistic Prototype Calibration of Vision-language Models for Generalized Few-shot Semantic Segmentation**[[paper]](https://arxiv.org/pdf/2506.22979)[[code]](https://github.com/jliu4ai/FewCLIP)
4. [2025 ICCV] **Few-Shot Pattern Detection via Template Matching and Regression**[[paper]](https://arxiv.org/pdf/2508.17636)
5. [2025 ICCV] **Unknown Text Learning for CLIP-based Few-Shot Open-set Recognition**
6. [2025 ICCV] **Text Augmented Correlation Transformer For Few-shot Classification & Segmentation**[[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Nandam_Text_Augmented_Correlation_Transformer_For_Few-shot_Classification__Segmentation_CVPR_2025_paper.pdf)
7. [2025 CVPR] **UNEM: UNrolled Generalized EM for Transductive Few-Shot Learning**[[paper]](https://arxiv.org/pdf/2412.16739)[[code]](https://github.com/ZhouLong0/UNEM-Transductive)


### Training-Free：
1. [2024 CVPR] **Clip-diy: Clip dense inference yields open-vocabulary semantic segmentation for-free** [[paper]](https://arxiv.org/pdf/2309.14289)[[code]](https://github.com/wysoczanska/clip-diy)
2. [2024 CVPR] **Training-Free Open-Vocabulary Segmentation with Offline Diffusion-Augmented Prototype Generation** [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10655445&tag=1) [[code]](https://github.com/aimagelab/freeda)
3. [2024 ECCV] **Diffusion Models for Open-Vocabulary Segmentation** [[paper]](https://arxiv.org/pdf/2306.09316) [[code]](https://github.com/karazijal/ovdiff)
4. [2024 ECCV] **ClearCLIP: Decomposing CLIP Representations for Dense Vision-Language Inference** [[paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06346.pdf)👌 [[code]](https://github.com/mc-lan/ClearCLIP)
>CLIP产生了具有错误分割区域的嘈杂分割图，去除残差连接、实现自注意力和丢弃前馈网络。ClearCLIP 始终如一地生成更清晰、更准确的分割图
5. [2024 ECCV] **SCLIP: Rethinking Self-Attention for Dense Vision-Language Inference** [[paper]](https://arxiv.org/pdf/2312.01597)👌 [[code]](https://github.com/wangf3014/SCLIP)
>CLIP的分割性能不佳是由斑块表示的空间错位引起的，而不是无法提取密集的视觉特征,问题出在CLIP的自注意力模块,使用 CSA 模块代替 CLIP 视觉编码器中的原始自注意力块
6. [2024 ECCV] **Pay Attention to Your Neighbours: Training-Free Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2404.08181) [[code]](https://github.com/sinahmr/NACLIP)
7. [2024 ECCV] **Proxyclip: Proxy attention improves clip for open-vocabulary segmentation** [[paper]](https://arxiv.org/pdf/2408.04883) [[code]](https://github.com/mc-lan/ProxyCLIP?tab=readme-ov-file) 
8. [2024 ECCV] **Explore the Potential of CLIP for Training-Free Open Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2407.08268) [[code]](https://github.com/leaves162/CLIPtrase)
9. [2024 ICLR] **A Hard-to-Beat Baseline for Training-free CLIP-Based Adaptation** [[paper]](https://openreview.net/forum?id=Js5PJPHDyY) [[code]](https://github.com/mrflogs/ICLR24)
10. [2024 arXiv] **CLIPer: Hierarchically Improving Spatial Representation of CLIP for Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2411.13836) [[code]](https://github.com/linsun449/cliper.code?tab=readme-ov-file)
11. [2025 CVPR] **LPOSS: Label Propagation Over Patches and Pixels for Open-vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2503.19777) [[code]](https://github.com/vladan-stojnic/LPOSS)
12. [2025 CVPR] **ResCLIP: Residual Attention for Training-free Dense Vision-language Inference** [[paper]](https://arxiv.org/pdf/2411.15851)👌 [[code]](https://github.com/yvhangyang/ResCLIP?tab=readme-ov-file)
>残差互相关自注意力 （RCS） 和语义反馈细化 （SFR） 模块。这两个模块可以纠正最后一层的注意力，以捕获特定类的特征和局部一致性，从而改进密集视觉语言预测任务的CLIP模型。
13. [2025 CVPR] **Distilling Spectral Graph for Object-Context Aware Open-Vocabulary Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Kim_Distilling_Spectral_Graph_for_Object-Context_Aware_Open-Vocabulary_Semantic_Segmentation_CVPR_2025_paper.pdf) [[code]](https://github.com/MICV-yonsei/CASS)
14. [2025 CVPR] **Cheb-GR: Rethinking k-nearest neighbor search in Re-ranking for Person Re-identification** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_Cheb-GR_Rethinking_K-nearest_Neighbor_Search_in_Re-ranking_for_Person_Re-identification_CVPR_2025_paper.pdf) [[code]](https://github.com/Jinxi-Yang-WHU/Fast-GCR.git) [[note]](本文提到的很多re-ranking的技术就是对直接计算的相似度矩阵进行更新，前面公式搞了一大堆，最后就是一个特征传播。)
15. [2025 CVPR] **ITACLIP: Boosting Training-Free Semantic Segmentation with Image, Text, and Architectural Enhancements** [[paper]](https://openaccess.thecvf.com/content/CVPR2025W/PixFoundation/papers/Aydin_ITACLIP_Boosting_Training-Free_Semantic_Segmentation_with_Image_Text_and_Architectural_CVPRW_2025_paper.pdf) [[code]](https://github.com/m-arda-aydn/ITACLIP)
16. [2025 CVPR] **Search and Detect: Training-Free Long Tail Object Detection via Web-Image Retrieval** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Sidhu_Search_and_Detect_Training-Free_Long_Tail_Object_Detection_via_Web-Image_CVPR_2025_paper.pdf) [[code]](https://github.com/Mankeerat/SearchDet)
17. [2025 ICCV] **LUDVIG: Learning-free Uplifting of 2D Visual features to Gaussian Splatting scene** [[paper]](https://arxiv.org/pdf/2410.14462#page=17.85) [[code]](https://github.com/naver/ludvig)
18. [2025 ICCV] **WildSeg3D: Segment Any 3D Objects in the Wild from 2D Images** [[paper]](https://arxiv.org/pdf/2503.08407)[[code]](https://github.com/Ethan16162/WildSeg3D)
19. [2025 ICCV] **Harnessing Vision Foundation Models for High-Performance, Training-Free Open Vocabulary Segmentation** [[paper]](https://arxiv.org/pdf/2411.09219) [[code]](https://github.com/YuHengsss/Trident)
20. [2025 ICCV] **E-SAM: Training-Free Segment Every Entity Model** [[paper]](https://arxiv.org/pdf/2503.12094)
21. [2025 ICCV] **ReME: A Data-Centric Framework for Training-Free Open-Vocabulary Segmentation** [[paper]](https://arxiv.org/pdf/2506.21233) [[code]](https://github.com/xiweix/ReME)
22. [2025 ICCV] **CCL-LGS: Contrastive Codebook Learning for 3D Language Gaussian Splatting** [[paper]](https://arxiv.org/pdf/2505.20469) [[code]](https://epsilontl.github.io/CCL-LGS/)
23. [2025 ICCV] **Auto-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2312.04539)[[code]](https://github.com/ozzyou/AutoSeg)
24. [2025 ICCV] **Understanding Personal Concept in Open-Vocabulary Semantic Segmentation**
25. [2025 ICCV] **Training-Free Class Purification for Open-Vocabulary Semantic Segmentation**[[paper]](https://arxiv.org/pdf/2508.00557)
26. [2025 ICCV] **DIH-CLIP: Unleashing the Diversity of Multi-Head Self-Attention for Training-Free Open-Vocabulary Semantic Segmentation**
27. [2025 ICCV] **Correspondence as Video: Test-Time Adaption on SAM2 for Reference Segmentation in the Wild**[[paper]](https://arxiv.org/pdf/2508.07759)[[code]](https://github.com/wanghr64/cav-sam)
28. [2025 ICCV] **Feature Purification Matters: Suppressing Outlier Propagation for Training-Free Open-Vocabulary Semantic Segmentation**[[paper]](https://kimsure.github.io/images/files/iccv25_sfp_full.pdf)[[code]](https://github.com/Kimsure/SFP)
29. [2025 ICCV] **Plug-in Feedback Self-adaptive Attention in CLIP for Training-free Open-Vocabulary Segmentation**[[paper]](https://arxiv.org/pdf/2508.20265)
30. [2025 ICCV] **Test-Time Retrieval-Augmented Adaptation for Vision-Language Models**[[code]](https://github.com/xinqi-fan/TT-RAA)
31. [2025 ICCV] **ConformalSAM: Unlocking the Potential of Foundational Segmentation Models in Semi-Supervised Semantic Segmentation with Conformal Prediction**[[paper]](https://arxiv.org/pdf/2507.15803)
32. [2025 ICCV] **Text-guided Visual Prompt DINO for Generic Segmentation**[[paper]](https://arxiv.org/pdf/2508.06146)[[code]](https://github.com/WeChatCV/WeVisionOne)
33. [2025 ICCV] **SCORE: Scene Context Matters in Open-Vocabulary Remote Sensing Instance Segmentation**[[paper]](https://arxiv.org/pdf/2507.12857)[[code]](https://github.com/HuangShiqi128/SCORE)
34. [2025 ICCV] **Images as Noisy Labels: Unleashing the Potential of the Diffusion Model for Open-Vocabulary Semantic Segmentation**
35. [2025 arXiv] **Self-Calibrated CLIP for Training-Free Open-Vocabulary Segmentation** [[paper]](https://arxiv.org/pdf/2411.15869) [[code]](https://github.com/SuleBai/SC-CLIP?tab=readme-ov-file)
36. [2025 arXiv] **Test-Time Adaptation of Vision-Language Models for Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2505.21844v1) [[code]](https://github.com/dosowiechi/MLMP?tab=readme-ov-file)
37. [2025 arXiv] **FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/abs/2504.10487) [[code]](https://github.com/yasserben/FLOSS)
38. [2025 arXiv] **TextRegion: Text-Aligned Region Tokens from Frozen Image-Text Models** [[paper]](https://arxiv.org/pdf/2505.23769) [[code]](https://github.com/avaxiao/TextRegion)
39. [2025 arXiv] **A Survey on Training-free Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2505.22209)

### 老师提供暂存：
1. [2025 arXiv] **POT: Prototypical Optimal Transport for Weakly Supervised Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_POT_Prototypical_Optimal_Transport_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2025_paper.pdf)
> 弱监督病理图像的语义分割
2. [2025CVPR] **Multi-Label Prototype Visual Spatial Search for Weakly Supervised Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Duan_Multi-Label_Prototype_Visual_Spatial_Search_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2025_paper.pdf)
> 弱监督病理图像的语义分割
3. [2024 arXiv]**Toward Modality Gap: Vision Prototype Learning for Weakly-supervised Semantic Segmentation with CLIP** [[paper]](https://arxiv.org/pdf/2412.19650)
4. [2025CVPR]**Prompt Categories Cluster for Weakly Supervised Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2025W/eLVM/papers/Wu_Prompt_Categories_Cluster_for_Weakly_Supervised_Semantic_Segmentation_CVPRW_2025_paper.pdf)
5. [2025 arXiv]  **2025-NIPS-Disentangling CLIP for Multi-Object Perception** [[paper]](https://arxiv.org/html/2502.02977v3)
6. [2021 ICLR] **A Trainable Optimal Transport Embedding for Feature Aggregation and its Relationship to Attention**[[paper]](https://arxiv.org/pdf/2006.12065)
7.[2025 ICCV] **Interpretable point cloud classification using multiple instance learning**[[paper]]()


### 3D点云处理和视觉-语言模型
1. [2025-ICCV] **Exploiting Vision Language Model for Training-Free 3D Point Cloud OOD Detection via Graph Score Propagation**
2. [2025-ICCV] **Describe, Adapt and Combine: Empowering CLIP Encoders for Open-set 3D Object Retrieval**
3. [2025-ICCV] **Partially Matching Submap Helps: Uncertainty Modeling and Propagation for Text to Point Cloud Localization**
4. [2025-ICCV] **Domain-aware Category-level Geometry Learning Segmentation for 3D Point Clouds**
5. [2025-CVPR] **Generalized Few-shot 3D Point Cloud Segmentation with Vision-Language Model**
6. [2025-arXiv] **Exploiting Vision Language Model for Training-Free 3D Point Cloud OOD Detection via Graph Score Propagation**  [[paper]](https://arxiv.org/html/2506.22375v1)
7. [2025-ICLR] **MULTIMODALITY HELPS FEW-SHOT 3D POINT CLOUD SEMANTIC SEGMENTATION** [[paper]](https://arxiv.org/pdf/2410.11414)
8. [2025-ICML] **SMART-PC: Skeletal Model Adaptation for Robust Test-Time Training in Point Clouds**
9. [2025-CVPR] **Point-Cache: Test-time Dynamic and Hierarchical Cache for Robust and Generalizable Point Cloud Analysis**
10. [2025-CVPR] **Purge-Gate: Efficient Backpropagation-Free Test-Time Adaptation for Point Clouds via Token Purging**
11. [2025-ICCV] **Describe, Adapt and Combine: Empowering CLIP Encoders for Open-set 3D Object Retrieval**

