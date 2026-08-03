DigitalSignalProcessing138(2023)104054
Contentslistsavailableat ScienceDirect
Digital Signal Processing
journal homepage: www.elsevier.com/locate/dsp

# Low-light image enhancement based on sharpening-smoothing image filter

Y. Demir, N.H. Kaplan
Erzurum Technical University, Electrical and Electronics Engineering Department, Erzurum, 25050, Turkiye

## Abstract

Low-light images suffer from poor visibility, severe noise, low contrast, and low brightness. To overcome these issues, many image enhancement methods have been proposed. Few techniques solve these problems simultaneously. This paper presents a low-light image enhancement method. The proposed method decomposes the input image into approximation and detail sub-images of the V component using a Smoothing-Sharpening Image Filter (SSIF) applied to the HSV (Hue, Saturation, Value) color space. After the decomposition process, Contrast-Limited Adaptive Histogram Equalization (CLAHE) is applied to the final approximation image to provide higher contrast. The detail sub-images are amplified and added to the enhanced approximation image to reconstruct the enhanced V component. Finally, inverse HSV transform is applied to the enhanced V component and H, S components to obtain the enhanced image. The experimental results show that the proposed method provides better visual quality and more natural colors than the compared state-of-the-art methods.

Keywords: Low-light image enhancement, Smoothing-sharpening image filter, Contrast limited adaptive histogram equalization, Multiscale decomposition

©2023 Elsevier Inc. All rights reserved.

## 1. Introduction

Images captured in low-light environments suffer from visual quality degradation, low contrast, poor visibility, and intensive noise. Despite the advanced image capture devices and special shooting techniques, image distortions cannot be removed entirely. In low-light conditions, with insufficient light reaching the camera sensors, avoiding the noise is quite difficult [1]. Low-light image enhancement prevents this issue in later image processing steps. Researchers have proposed many methods to enhance low-light images. Thus, better inputs are provided for further image processing tasks.

Researchers have made numerous innovations in low-light image enhancement methods in recent decades. The methods that can be considered as the basis of image enhancement in low-light directly increase the general illumination information of the image. Low-light image enhancement methods are divided into three categories. The first category is histogram-based methods, in which the methods stretch the dynamic range of the image [2,3]. These methods generally cause noise amplification and undesired lighting. To overcome these issues, many methods to improve the HE performance by variational regularization on the histogram are proposed. Even though regularization has improved the performance of the HE methods, the desired results can not be obtained. These types of methods cause illumination problems, over-enhancement, and under-enhancement. In the second category, namely Retinex theory-based methods [4], the image is decomposed into reflection and illumination layers. Multi-scale Retinex (MSR) [5] and single-scale Retinex (SSR) [6] are early approaches based on Retinex theory. These methods, considered the basis of the Retinex theory, suffer from unnaturalness and over-enhancement. Different filters [5–7] have been proposed to decompose images, and different priors [8–12] are developed on the separated reflection and illumination layers. However, these methods can not provide high visual quality and frequently suffer from intensive noise. Moreover, the results are not robust enough and may cause over and under-enhancement.

The final category is learning-based methods. Recently, with the rapid development in learning based methods, many CNN-based methods have been proposed for low-light problems. Learning-based methods with strong representation capability are significantly developed to improve the degradation [13–21]. These methods have complex models designed based on learning and mapping the paired images, which are well-prepared as low and normal-light. In addition, loss functions are unsuitable for human visual perception and may cause undesired results. Residual noise, over-enhanced contrast, and detail loss can be examples of undesired effects.

This paper proposes a low-light image enhancement method based on the Smoothing-Sharpening Image Filter (SSIF) [22].

Smoothing and sharpening are used in image processing as two fundamental operations. In general, smoothing is used to reduce noise, while sharpening is used to enhance details [23]. Therefore, to obtain an improved result, image sharpening is significant. However, directly sharpening the image will amplify the noise, as well. Consequently, a multi-scale decomposition scheme based on SSIF is proposed, similar to the decomposition schemes of [24,25]. The output of the SSIF filter is interpreted as the base layer, and the difference between the original image and the filtered one is called the detail layer. The SSIF filter is applied to the base layer to obtain a multi-scale decomposition. To construct further detail levels, the adjacent base layers are subtracted. By multi-scale SSIF decomposition, the original image's high and low frequency components are separated. After certain decomposition levels, Contrast Limited Adaptive Histogram Equalization (CLAHE) is applied to the final base layer. By applying CLAHE to the final base layer instead of the whole image, higher visibility with lower noise amplification is expected. Moreover, the detail layers are amplified by an arbitrary coefficient. In this way, the detail level of the image is increased. Obtained new base layer and detail layers are summed up to get the final enhanced image. In summary, contributions of the paper are given below:

- A multi-scale decomposition scheme for Sharpening Smoothing Image Filter (SSIF) is proposed. In this way, the high and low frequency components of the image are separated.
- The visibility of the base layer is improved by applying CLAHE, and the details are enhanced by amplifying the detail layers. In this way, higher visibility with a lower noise level is achieved.

The rest of the paper is organized as follows. In Section 2, background information for SSIF and CLAHE is given. The proposed multi-scale decomposition and enhancement methodology are described in Section 3. The experimental results are discussed in Section 4. Finally, the paper is concluded in Section 5.

## 2. Related work

This section provides background information about SSIF and CLAHE used in the proposed methodology.

### 2.1. Guided smoothing-sharpening filter

The guided SSIF can be given as follows [22]:

J_k(q) = μ_k + sign(φ_k) α_k (G_k(q) − v_k)  (1)

Here, J is the filtered image, I is the original image, and G is the guidance image. μ_k is the mean of the input image, and v_k is mean of the guidance image within the patch k. α_k is the sample patch covariance of the original and guidance images. Here, α_k is a positive parameter. More specifically, the negative likelihood can be expressed as [22]:

−log p({J_k(q)}|α_k) = (1 / (2τ²N)) Σ_{q∈Ω_k} (I_k(q) − J_k(q))²  (2)

where τ² is the patch variance of the filtered image, which is set to τ=1, the symbol Ω represents the set of pixels within the patch. N is the number of pixels, ς² is patch variance of the guided image and is defined as:

ς²_k = (1/N) Σ_{q∈Ω_k} (G_k(q) − v_k)²  (3)

Using the Generalized Gamma distribution as the prior, negative log-posterior is obtained as cost function D(α_k).

D(α_k) = −log p({I_k(q)}|α_k) − log p(α_k)
        = ς²_k / (2α_k²) − |φ_k| α_k + α_k² / (2θ²) − η log α_k  (4)

where θ>0 is the scale parameter and η≥0 is the parameter to control the shape of the distribution. Solving ∂D/∂α_k=0 by letting ε=1/θ², η=κε, the results can be re-arranged as follows [22]:

α_k = |φ_k|/2 · ( ς²_k/(ς²_k+ε) ) + sqrt{ (|φ_k|/2)² (ς²_k/(ς²_k+ε))² + 4κε · ς²_k/(ς²_k+ε) }  (5)

### 2.2. Contrast limited adaptive histogram equalization

Classical HE improves contrast but is not limited to a specific region, which may result in loss of information during the image enhancement process. CLAHE [2] is used to avoid information losses. In this method, the image is divided into sub-blocks. For instance, if the image size is 512X512, the input image is usually split into 64 sub-blocks to get a good statistical estimation [26]. In this case, the histogram is limited to small blocks, and the noise can be amplified. A contrast limit is used to avoid this issue. The main problem in CLAHE is the contrast of the sub-blocks reaching the maximum. The crop limit is used to limit the contrast. The procedure can be given as follows:

β = 1 + (M·N/100) · (s_max − 1) · (α/100)  (6)

Here, β is the crop limit, M and N are the width and the height of the input image, respectively. α is the crop factor and s_max is maximum value of the transfer function. The crop factor takes values in the [0, 100] range. For α=0, there is no change in the pixels. It is seen that for the maximum value of α, the crop limit is equal to s_max.

## 3. Proposed method

The proposed method first decomposes the original image into high and low frequency components by a multi-scale SSIF decomposition. This way, the input image's details, and rough information are separated. After the decomposition process, CLAHE is applied to the low frequency component to obtain higher visibility. The high frequency components are amplified by an arbitrary coefficient to obtain a higher level of detail. Finally, the resulting low and high frequency components are combined to obtain the final enhanced image.

### 3.1. Multi-scale decomposition with SSIF

The SSIF uses four parameters during the filtering process, namely patch radius (r), Kappa (κ), Epsilon (ε) and Scale (s) [22]. Setting a larger (r) value of the filter produces a solid smoothing result. Changing the (r) value impacts the variance of each pixel position and produces a change in the sharpening gain. The value of κ controls the smoothing/sharpening gain of the filter. If the values of κ>1, the filter interprets as a sharpening filter, whereas values κ<1 show the effect of a smoothing filter. In the case of κ=1, there is no filtering. Setting a larger value of regularization parameter ε produces washed-out results. Scale (s) is a user-defined parameter. The filter output for an input image I can be given as follows:

S1[I] = SSIF(I)  (7)

Here, SSIF(.) is the SSIF operator, and S1 can be called the first approximation layer. For the multi-scale decomposition of SSIF, a similar manner described in [25] is followed. In order to obtain further levels of approximation layer, the filtering outputs are filtered with SSIF repeatedly as:

Sj[I] = SSIF(Sj−1)  (8)

where j is the decomposition level and S0=I. The detail images at decomposition level j can be obtained with:

Dj[I] = Sj−1[I] − Sj[I]  (9)

For L levels of decomposition, the original image can be reconstructed as:

I = Σ_{j=1}^{L} Dj[I] + SL[I]  (10)

The flowchart for two levels of decomposition and reconstruction is given in Fig. 1.

### 3.2. Low-light enhancement scheme

The low-light image algorithm can be summarized as follows (see Table 1):

- The input image is transformed from RGB color space to HSV color space. All subsequent operations are applied to the V subband until the transformation of HSV color space to RGB color space.
- The proposed multi-scale SSIF is applied to the V component for two levels of decomposition to obtain the detail images and final approximation image.
- CLAHE is applied to the final approximation image to provide higher visibility on the enhanced image.
- Detail images of each decomposition level are amplified with a predefined enhancement coefficient.
- The enhanced V component is reconstructed by simply adding the amplified detail images and CLAHE output.
- Inverse HSV transform is applied to the H and S components of the input image and enhanced V component to obtain the final enhanced image.

According to the proposed enhancement algorithm, the enhanced image can be given as:

IE = CLAHE[SL] + Σ_{j=1}^{L} ω_j Dj[I]  (11)

Here, ω_j is an arbitrary coefficient. CLAHE[.] is the CLAHE operator, and IE is the final enhanced image.

The overall enhancement procedure is given in Fig. 2.

**Table 1**
Pseudo-code for the proposed enhancement method.

**Input**
- I: low-light input image
- r: patch radius
- (κ): Kappa
- (ε): Epsilon
- s: Scale
- β: clip limit

**Main**
1. Apply HSV transform to I
2. Two level decomposition to V subband with (7)
3. Apply CLAHE to residual image with (6)
4. Weighting detail images with coefficient with (11)
5. Apply inverse HSV transform to IE

**Output**
- Final enhanced result

## 4. Experimental results

### 4.1. Dataset

Since the proposed method is independent of image format and size, it has been applied to a dataset covering various image attributes and properties in low-light images. Datasets with standard ones used in previous studies (NPE [7], LIME [10], MEF [27], DCIM [28], VV) are used to test the images. Moreover, all test sets [20] are used to compare the performance of the methods on the test set. The NPE dataset consists of 84 images and includes various images such as night and under-exposure daytime images. The LIME dataset consists of 10 images and includes extra dark images. MEF dataset consists of 17 images and includes multi-exposure images. The DICM dataset consists of 64 images and includes various multi-exposure daily images. VV dataset consists of 24 images and includes under-exposure daily images. VE-LOL [29] is used for evaluating low-level and high-level vision, an advanced version of LOL [30]. In addition, the VE-LOL dataset is used to evaluate synthetic and real images comprehensively. VE-LOL-H includes 10940 real unpaired images (6940 for training and validation and 4000 for testing), and we use 6940 training and validation images. VE-LOL-L consists of VE-LOL-Syn, which includes 1000 synthetic paired images, and VE-LOL-Cap, which includes 1500 real paired images. We use 1000 synthetic images (VE-LOL-Syn) and 500 real images (VE-LOL-Cap training and test images). In addition, Part2 Subset [15], which is also used in previous studies, is used to compare the performance of different methods for full-reference metrics quantitatively. Part2 subset, which consists of 229 images, has multi-exposure sequences for each image. Real-world images are an indispensable part of low-light enhancement. To this end, Berkeley Deep Driving (BDD-100k) [31] dataset is used to prove our method's effectiveness on real-world images. The low-light image enhancement process is performed on an 11th Gen Intel(R) Core(TM) i7-11800H CPU, and the processing time varies depending on the size of the input image.

### 4.2. Implementation studies

Several implementation studies are conducted to demonstrate each parameter's effectiveness for the proposed method. In order to determine the optimum parameter values for the SSIF filter, the effect of all parameters on the PSNR value is investigated. For this purpose, 27 images are chosen with different features in terms of exposure, contrast, and visibility. Finally, the mean PSNR value of 27 images is calculated for each parameter. The κ has a significant effect on enhancement results. The impact of remaining parameters (r), ε, and s have a negligible impact on PSNR in experimental studies. The effect of the κ parameter on the PSNR value is given in Fig. 3. All parameters having the highest PSNR values are determined as optimum values. The parameters of the proposed method are set as κ=2, r=19, ε=0.001, and s=0.1.

### 4.3. Comparison with state-of-the-art methods

#### 4.3.1. Visual comparison

First, the visual quality of the proposed method is compared with recently proposed methods. Visual results are demonstrated in Fig. 4 and Fig. 5 where Fig. 4a and Fig. 5a shows the original low-light image, and Fig. 4b-g and Fig. 5b-g are the enhanced images by: SRIE [9], LIME [10], Robust Retinex [11], DeepUPE [18], EnlightenGAN [20], and Zero-DCE [19] respectively. Fig. 4h and Fig. 5h show the results obtained by the proposed method. To see the results clearly, some parts of the images are zoomed in and given at the bottom of each image. In Fig. 4b, SRIE has a little noise in some parts of the sky, and generally, Eiffel Tower is darker than other methods. LIME has intensive noise. In addition, the colors of the iron structure have turned into red color in the green box, as seen in Fig. 4c. In Fig. 4d, Robust Retinex caused some edge information loss and smoothed some regions of the building, as seen in the red box. In Fig. 4e, DeepUPE has no noise but can not provide sufficient low-light enhancement. EnlightenGAN leads to artifacts that make the glaring results as seen in both zoomed boxes, in Fig. 4f. Zero-DCE has severe noise in the sky region, as seen in the green box, and causes unnatural colors on the buildings, as seen in Fig. 4g. Fig. 4h demonstrates that the proposed method has effective results in terms of visual quality, visibility, and naturalness. In Fig. 5, SRIE provides significant enhancement, as seen in both zoomed areas. In Fig. 5b, as seen in the red box, tree branches are visible in the SRIE method but cause noise and distortions simultaneously. LIME has severe noise, as seen in the red box, and halo effects are present around the street lamp. Robust Retinex has noise in some regions of the sky, and there are information losses because of the smoothing. In Fig. 5e, DeepUPE performs well in terms of noise and artifacts. Still, tree branches are not visible, as seen in the red box, and amplify the illumination excessively on the illuminated regions. EnlightenGAN has intensive noise, as seen in the sky region, and causes excessive brightness overall. Zero-DCE improves some regions of the image better than former methods, such as the wall on the left side but suffers from intensive noise and redundant brightness. In the proposed method, tree branches are visible, as seen in the red box in Fig. 5h. In addition, there is no noise, halo effects, or color distortions. Low-light enhancement methods need to be adaptable according to a variety of conditions. Therefore, we apply the proposed method to daytime and person images. In Fig. 6, the proposed method provides the best color preservation, as seen in the wood and flower details in the green box. Moreover, the proposed method provides better detail enhancement on the wall region, demonstrated in the red box. In Fig. 7, Zero-DCE achieves the best result in background illumination enhancement, whereas the proposed method reaches the best result with the least color distortion in the face region. In terms of all these features, the proposed method outperforms the state-of-the-art methods.

#### 4.3.2. Full-referenced image quality assessment

Peak Signal-to-Noise Ratio (PSNR), Structural Similarity (SSIM) [32], and low-light image enhancement quality assessment (LIEQA) [33] metrics are adapted to quantitatively compare the performance of all methods on the Part2 subset and VE-LOL-L. The higher PSNR value indicates higher image quality, and the higher SSIM value indicates more similarity between the two images. The higher LIEQA value indicates higher low-light enhancement performance. In Table 2, the results in red represent the best result, whereas the results in blue represent the second-best result. On the Part2 subset, the proposed method achieves the best values in all cases. Zero-DCE reaches the second-best PSNR value after the proposed method. Zero-DCE and Enlighten-GAN achieve the second-best value in the SSIM value. LIME has the second-best performance in terms of the LIEQA metric. The proposed method provides better restored global illumination, repaired structural details, and stretched contrast in light of full reference evaluation metric results. On the VE-LOL-L dataset, Enlighten-GAN reaches the best PSNR and SSIM values, Zero-DCE reaches the second-best PSNR value, and DeepUPE reaches the second-best SSIM value. Zero-DCE achieves the best LIEQA value, while SRIE achieves the second-best LIEQA value.

**Table 2**
Full reference metrics comparisons for benchmark datasets.

| Dataset | Metric | SRIE[9] | LIME[10] | RobustRetinex[11] | DeepUPE[18] | EnlightenGAN[20] | Zero-DCE[19] | Proposed |
|---|---|---|---|---|---|---|---|---|
| Part2 Subset | PSNR | 14.416 | 16.178 | 15.197 | 14.426 | 16.213 | 16.572 | 16.713 |
| Part2 Subset | SSIM | 0.549 | 0.573 | 0.544 | 0.691 | 0.591 | 0.594 | 0.735 |
| Part2 Subset | LIEQA | 0.211 | 0.317 | 0.183 | 0.151 | 0.198 | 0.131 | 0.364 |
| VE-LOL-L | PSNR | 13.601 | 13.479 | 14.091 | 13.530 | 14.858 | 14.271 | 13.140 |
| VE-LOL-L | SSIM | 0.403 | 0.394 | 0.394 | 0.409 | 0.449 | 0.375 | 0.388 |
| VE-LOL-L | LIEQA | 0.104 | 0.078 | 0.081 | 0.099 | 0.101 | 0.108 | 0.076 |

#### 4.3.3. No-reference image quality assessment

NIQE [34] is a well-known image quality assessor used for images without ground truth. The NIQE algorithm is based on quality features extracted from real natural images and uses the natural scene statistic (NSS) model. A lower NIQE value means that the corresponding image has better visual quality. A no-reference low-light image enhancement evaluation (NLIEE) [35] index is used to perform a more objective evaluation. NLIEE considers light enhancement, color comparison, noise measurement, and structure similarity. The lower NLIEE value indicates better low-light enhancement performance. The NIQE and NLIEE results obtained for different datasets (DICM, LIME, MEF, VV, NPE, and VE-LOL-H) are reported in Table 3. Robust Retinex and DeepUPE achieve the best NIQE result on the DICM dataset, which has multi-exposure images, while Zero-DCE achieves the best NLIEE value. In addition, the proposed method achieves the best NIQE and NLIEE values following these methods. Since the LIME dataset has extra dark images, EnlightenGAN achieves the best NIQE result, and DeepUPE reaches the best NLIEE value on the LIME dataset, as expected. Zero-DCE achieves the best NIQE and NLIEE result on the MEF dataset. The proposed method achieves the best result of two metrics on the VV and NPE datasets, which generally have under-exposure images. The best average NIQE and NLIEE scores are obtained by the proposed method. The proposed method reaches the second-best result on the VE-LOL-H dataset. Thus, the proposed method provides superior performance compared to state-of-the-art techniques. According to the NIQE and NLIEE evaluation metrics, the proposed method performs better on under-exposure images.

**Table 3**
No reference metrics comparisons for benchmark datasets.

| Dataset | Metric | SRIE[9] | LIME[10] | RobustRetinex[11] | DeepUPE[18] | EnlightenGAN[20] | Zero-DCE[19] | Proposed |
|---|---|---|---|---|---|---|---|---|
| DICM | NIQE | 3.899 | 3.846 | 3.238 | 3.238 | 3.570 | 3.309 | 3.282 |
| DICM | NLIEE | 49.686 | 48.730 | 51.895 | 42.166 | 52.414 | 40.213 | 41.712 |
| LIME | NIQE | 3.788 | 4.155 | 4.108 | 3.789 | 3.719 | 4.045 | 3.870 |
| LIME | NLIEE | 49.541 | 50.883 | 52.957 | 48.309 | 53.765 | 50.636 | 50.800 |
| MEF | NIQE | 3.475 | 3.720 | 4.204 | 3.860 | 3.232 | 3.123 | 3.503 |
| MEF | NLIEE | 44.039 | 44.058 | 46.102 | 41.806 | 46.094 | 39.907 | 40.312 |
| VV | NIQE | 2.850 | 2.489 | 2.623 | 2.237 | 2.581 | 2.794 | 1.965 |
| VV | NLIEE | 48.092 | 49.863 | 48.346 | 42.367 | 37.229 | 36.727 | 32.734 |
| NPE | NIQE | 3.986 | 4.268 | 4.047 | 3.589 | 4.113 | 3.734 | 3.577 |
| NPE | NLIEE | 54.723 | 56.210 | 58.177 | 52.285 | 48.027 | 52.149 | 51.949 |
| All Test Set | NIQE | 3.650 | 3.629 | 3.670 | 3.491 | 3.385 | 3.506 | 3.358 |
| All Test Set | NLIEE | 50.791 | 51.618 | 53.381 | 47.249 | 48.208 | 45.955 | 43.707 |
| VE-LOL-H | NIQE | 3.538 | 3.954 | 4.014 | 3.327 | 2.592 | 2.581 | 3.625 |
| VE-LOL-H | NLIEE | 38.356 | 36.984 | 39.747 | 36.949 | 47.288 | 29.912 | 35.063 |

### 4.4. Application on real-world images

Daily images rather than aesthetic images are an indispensable part of the real world. The experiments are carried out using the BBD-100k dataset from real-world driving to show the advantage of the proposed method in practice. Random images from the BDD-100k dataset are selected as low-light images. Those images suffer from poor visibility and low brightness. Then, comparisons with state-of-the-art methods are carried out. In Fig. 8, SRIE has the lowest noise but is darker than the other methods. LIME suffers from intensive noise, especially in regions of light sources in the image. In addition, vehicle headlights cause artifacts, making halo effects in this method. Robust Retinex causes some detail information loss, as seen inside the bus and the region of asphalt in front of the bus. Fig. 8e demonstrates DeepUPE, which has weak low-light enhancement and poor visibility. EnlightenGAN suffers from high ISO noise, which causes undesired color distortion, as seen on the right side of the image. Zero-DCE has effective results in terms of visibility but suffers from artifacts. The proposed method achieves the best result in all these visual assessments, as seen in Fig. 8h.

To compare the performance of state-of-the-art methods, another real-world image is chosen, including the rear headlight in Fig. 9. SRIE provides better visibility for the road region but has artifacts, as seen in the sky region of the image. In Fig. 9c, the LIME method caused some information losses around the rear lights of vehicles. In addition, LIME has severe noise, which causes color distortions. Robust Retinex causes high ISO noise on the image. Moreover, Robust Retinex has artifacts and information losses because of over-smoothing. Fig. 9e demonstrates DeepUPE, which has undesired color changes, as seen at the bottom of the image. EnlightenGAN has intensive noise, which causes unnatural colors and distortions in the image. Similarly, Zero-DCE has severe noise, which causes undesired color changes, as seen in the sky region. The proposed method provides better visual quality and contrast enhancement. Fig. 10 demonstrates an image captured from a night drive inside a vehicle. SRIE provides good contrast enhancement but suffers from noise. In addition, SRIE has an undesired shadow effect on the crosswalk lines.

In Fig. 10c, LIME has artifacts that cause halo effects and distortions. Robust Retinex has natural colors and low noise, but some details can not be seen clearly because of smoothing, such as the windows of the building. DeepUPE has undesired illumination at the edges of the image. Furthermore, contrast enhancement is insufficient, such as trees in front of the building. EnlightenGAN increases the brightness but causes unnatural colors, as seen at the top left of the image. In Fig. 10g, Zero-DCE suffers from high ISO noise, which causes artificial colors in the image. The proposed method leads to more visible details, natural colors, and low noise in the image.

## 5. Conclusion

In this paper, a novel low-light enhancement method based on the smoothing-sharpening image filter is proposed. Unlike most state-of-the-art techniques, the proposed method has a simple framework and effective results. Deep learning based methods suffer from noise amplification and halo effects. In addition to this, traditional low-light enhancement methods have poor visibility. In low-light image enhancement, it is essential to simultaneously provide strong visibility and high quality. To this end, a framework is developed to achieve these aims. The proposed low-light enhancement algorithm uses a smoothing-sharpening filter thus, provides better edge enhancement without severe noise. In addition, the proposed framework provides better contrast enhancement with the CLAHE. Contrary to the former methods, the proposed method provides better visual quality and better visibility simultaneously. Furthermore, the proposed method prevents halo effects. Quantitative and qualitative validations on various low-light datasets have provided satisfactory evidence against state-of-the-art methods.

## CRediT authorship contribution statement

**Y. Demir:** Conceptualization, Data curation, Investigation, Methodology, Software, Writing – original draft. **N.H. Kaplan:** Methodology, Software, Supervision, Validation, Visualization, Writing – review & editing.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data availability

Data will be made available on request.

## References

[1] W. Yang, S. Wang, Y. Fang, Y. Wang, J. Liu, Band representation-based semi-supervised low-light image enhancement: bridging the gap between signal fidelity and perceptual quality, IEEE Trans. Image Process. 30 (2021) 3461–3473.

[2] S. Pizer, R. Johnston, J. Ericksen, B. Yankaskas, K. Muller, Contrast-limited adaptive histogram equalization: speed and effectiveness, in: [1990] Proceedings of the First Conference on Visualization in Biomedical Computing, IEEE Computer Society, Los Alamitos, CA, USA, 1990, pp.337–345.

[3] M. Abdullah-Al-Wadud, M.H. Kabir, M.A. Akber Dewan, O. Chae, A dynamic histogram equalization for image contrast enhancement, IEEE Trans. Consum. Electron. 53 (2007) 593–600.

[4] E.H. Land, The retinex theory of color vision, Sci. Am. 237 (1977) 108–129.

[5] D. Jobson, Z. Rahman, G. Woodell, A multiscale retinex for bridging the gap between color images and the human observation of scenes, IEEE Trans. Image Process. 6 (1997) 965–976.

[6] D. Jobson, Z. Rahman, G.A. Woodell, Properties and performance of a center/surround retinex, IEEE Trans. Image Process. 6 (1997) 451–462.

[7] S. Wang, J. Zheng, H.-M. Hu, B. Li, Naturalness preserved enhancement algorithm for non-uniform illumination images, IEEE Trans. Image Process. 22 (2013) 3538–3548.

[8] X. Fu, D. Zeng, Y. Huang, Y. Liao, X. Ding, J. Paisley, A fusion-based enhancing method for weakly illuminated images, Signal Process. 129 (2016) 82–96.

[9] X. Fu, D. Zeng, Y. Huang, X.-P. Zhang, X. Ding, A weighted variational model for simultaneous reflectance and illumination estimation, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[10] X. Guo, Y. Li, H. Ling, Lime: low-light image enhancement via illumination map estimation, IEEE Trans. Image Process. 26 (2017) 982–993.

[11] M. Li, J. Liu, W. Yang, X. Sun, Z. Guo, Structure-revealing low-light image enhancement via robust retinex model, IEEE Trans. Image Process. 27 (2018) 2828–2841.

[12] X. Ren, M. Li, W.-H. Cheng, J. Liu, Joint enhancement and denoising method via sequential decomposition, in: 2018 IEEE International Symposium on Circuits and Systems (ISCAS), 2018, pp.1–5.

[13] K.G. Lore, A. Akintayo, S. Sarkar, Llnet: a deep autoencoder approach to natural low-light image enhancement, Pattern Recognit. 61 (2017) 650–662.

[14] C. Chen, Q. Chen, J. Xu, V. Koltun, Learning to see in the dark, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

[15] J. Cai, S. Gu, L. Zhang, Learning a deep single image contrast enhancer from multi-exposure images, IEEE Trans. Image Process. 27 (2018) 2049–2062.

[16] W. Wang, C. Wei, W. Yang, J. Liu, Gladnet: low-light enhancement network with global awareness, in: 2018 13th IEEE International Conference on Automatic Face & Gesture Recognition (FG 2018), IEEE, 2018, pp.751–755.

[17] W. Ren, S. Liu, L. Ma, Q. Xu, X. Xu, X. Cao, J. Du, M.-H. Yang, Low-light image enhancement via a deep hybrid network, IEEE Trans. Image Process. 28 (2019) 4364–4375.

[18] R. Wang, Q. Zhang, C.-W. Fu, X. Shen, W.-S. Zheng, J. Jia, Underexposed photo enhancement using deep illumination estimation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

[19] C. Guo, C. Li, J. Guo, C.C. Loy, J. Hou, S. Kwong, R. Cong, Zero-reference deep curve estimation for low-light image enhancement, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

[20] Y. Jiang, X. Gong, D. Liu, Y. Cheng, C. Fang, X. Shen, J. Yang, P. Zhou, Z. Wang, Enlightengan: deep light enhancement without paired supervision, IEEE Trans. Image Process. 30 (2021) 2340–2349.

[21] Y. Wang, R. Wan, W. Yang, H. Li, L.-P. Chau, A. Kot, Low-light image enhancement with normalizing flow, in: Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, 2022, pp.2604–2612.

[22] G. Deng, F. Galetto, M. Al–nasrawi, W. Waheed, A guided edge-aware smoothing-sharpening filter based on patch interpolation model and generalized gamma distribution, IEEE Open J. Signal Process. 2 (2021) 119–135.

[23] R.C. Gonzalez, R.E. Woods, Digital Image Processing, 3rd ed., Pearson Education, 2009.

[24] N.H. Kaplan, I. Erer, Remote sensing image enhancement via robust guided filtering, in: Proc. 2019 9th International Conference on Recent Advances in Space Technologies (RAST), 2019, pp.447–450.

[25] N. Kaplan, I. Erer, Scale aware remote sensing image enhancement using rolling guidance, J. Vis. Commun. Image Represent. 80 (2021) 103315.

[26] A.M. Reza, Realization of the contrast limited adaptive histogram equalization (clahe) for real-time image enhancement, J. VLSI Signal Process. Syst. Signal Image Video Technol. 38 (2004) 35–44.

[27] K. Ma, K. Zeng, Z. Wang, Perceptual quality assessment for multi-exposure image fusion, IEEE Trans. Image Process. 24 (2015) 3345–3356.

[28] C. Lee, C. Lee, C.-S. Kim, Contrast enhancement based on layered difference representation, in: 2012 19th IEEE International Conference on Image Processing, 2012, pp.965–968.

[29] J. Liu, D. Xu, W. Yang, M. Fan, H. Huang, Benchmarking low-light image enhancement and beyond, Int. J. Comput. Vis. 129 (2021) 1153–1184.

[30] C. Wei, W. Wang, W. Yang, J. Liu, Deep retinex decomposition for low-light enhancement, arXiv preprint, arXiv:1808.04560, 2018.

[31] F. Yu, H. Chen, X. Wang, W. Xian, Y. Chen, F. Liu, V. Madhavan, T. Darrell, Bdd100k: a diverse driving dataset for heterogeneous multitask learning, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp.2636–2645.

[32] Z. Wang, A. Bovik, H. Sheikh, E. Simoncelli, Image quality assessment: from error visibility to structural similarity, IEEE Trans. Image Process. 13 (2004) 600–612.

[33] G. Zhai, W. Sun, X. Min, J. Zhou, Perceptual quality assessment of low-light image enhancement, ACM Trans. Multimed. Comput. Commun. Appl. (TOMM) 17 (2021) 1–24.

[34] A. Mittal, R. Soundararajan, A.C. Bovik, Making a "completely blind" image quality analyzer, IEEE Signal Process. Lett. 20 (2013) 209–212.

[35] Z. Zhang, W. Sun, X. Min, W. Zhu, T. Wang, W. Lu, G. Zhai, A no-reference evaluation metric for low-light image enhancement, in: 2021 IEEE International Conference on Multimedia and Expo (ICME), IEEE, 2021, pp.1–6.

---

**Nur Huseyin Kaplan** received his B. Sc., M. Sc, and Ph.D degrees in Electronics and Telecommunication Engineering from Istanbul Technical University, Turkey. He worked for TURKSAT A.S. as a specialist between 2005-2015. Since 2018, he is an Associate Professor at Electrical and Electronics Engineering Department with Erzurum Technical University, Turkey, where he teaches undergraduate/graduate level courses about signal and image processing. His primary research interests include digital signal and image processing.

**Yasin Demir** received his B. Sc., degree in Electrical and Electronics Engineering form Karadeniz Technical University, Turkey. He received his M. Sc., degree in Electrical and Electronics Engineering form Erzurum Technical University, Turkey. He is a Research Assistant at Electrical and Electronics Engineering Department with Erzurum Technical University, Turkey. His primary research interests include digital signal and image processing.
