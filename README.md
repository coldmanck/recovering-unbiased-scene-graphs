# Recovering the Unbiased Scene Graphs from the Biased Ones
Official implementation of "Recovering the Unbiased Scene Graphs from the Biased Ones" (ACMMM 2021)

<div align="center">
    
[Meng-Jiun Chiou](http://coldmanck.github.io/), Henghui Ding, Hanshu Yan, Changhu Wang, [Roger Zimmermann](https://www.comp.nus.edu.sg/~rogerz/roger.html) and [Jiashi Feng](https://sites.google.com/site/jshfeng/home),<br>
"[Recovering the Unbiased Scene Graphs from the Biased Ones]()," to appear in [ACMMM 2021](https://2021.acmmm.org/).
</div>

## Introduction
Given input images, scene graph generation (SGG) aims to produce comprehensive, graphical representations describing visual relationships among salient objects. Recently, more efforts have been paid to the long tail problem in SGG; however, the imbalance in the fraction of missing labels of different classes, or reporting bias, exacerbating the long tail is rarely considered and cannot be solved by the existing debiasing methods. In this paper we show that, due to the missing labels, SGG can be viewed as a "Learning from Positive and Unlabeled data" (PU learning) problem, where the reporting bias can be removed by recovering the unbiased probabilities from the biased ones by utilizing label frequencies, i.e., the per-class fraction of labeled, positive examples in all the positive examples. To obtain accurate label frequency estimates, we propose Dynamic Label Frequency Estimation (DLFE) to take advantage of training-time data augmentation and average over multiple training iterations to introduce more valid examples. Extensive experiments show that DLFE is more effective in estimating label frequencies than a naive variant of the traditional estimate, and DLFE significantly alleviates the long tail and achieves state-of-the-art debiasing performance on the VG dataset. We also show qualitatively that SGG models with DLFE produce prominently more balanced and unbiased scene graphs.


## Models
The source code will be available at this repository soon. Stay tuned :)

## Citation

## Credits
