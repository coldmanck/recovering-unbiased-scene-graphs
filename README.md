# Recovering the Unbiased Scene Graphs from the Biased Ones (ACMMM 2021)
<!-- Official implementation of "Recovering the Unbiased Scene Graphs from the Biased Ones" (ACMMM 2021) -->

<div align="center">
    "A Simple yet Elegant Way to Remove Reporting Bias in Your Scene Graphs"
</div>

<div align="center">
    <img src="figs/motivation.jpg" width="500">    
    
[Meng-Jiun Chiou](http://coldmanck.github.io/), [Henghui Ding](https://henghuiding.github.io/), [Hanshu Yan](https://sites.google.com/view/hanshuyan/home), Changhu Wang, [Roger Zimmermann](https://www.comp.nus.edu.sg/~rogerz/roger.html) and [Jiashi Feng](https://sites.google.com/site/jshfeng/home),<br>
"[Recovering the Unbiased Scene Graphs from the Biased Ones]()," to appear in [ACMMM 2021](https://2021.acmmm.org/).
</div>

## Introduction
Given input images, scene graph generation (SGG) aims to produce comprehensive, graphical representations describing visual relationships among salient objects. Recently, more efforts have been paid to the long tail problem in SGG; however, the imbalance in the fraction of missing labels of different classes, or reporting bias, exacerbating the long tail is rarely considered and cannot be solved by the existing debiasing methods. In this paper we show that, due to the missing labels, SGG can be viewed as a "Learning from Positive and Unlabeled data" (PU learning) problem, where the reporting bias can be removed by recovering the unbiased probabilities from the biased ones by utilizing label frequencies, i.e., the per-class fraction of labeled, positive examples in all the positive examples. To obtain accurate label frequency estimates, we propose Dynamic Label Frequency Estimation (DLFE) to take advantage of training-time data augmentation and average over multiple training iterations to introduce more valid examples. Extensive experiments show that DLFE is more effective in estimating label frequencies than a naive variant of the traditional estimate, and DLFE significantly alleviates the long tail and achieves state-of-the-art debiasing performance on the VG dataset. We also show qualitatively that SGG models with DLFE produce prominently more balanced and unbiased scene graphs.

<div align="center">
    <img src="figs/DLFE.jpg" width="800">
</div>

## Visualizing Unbiased Scene Graphs
(Click to see enlarged images!)
<div align="center">
    <img src="figs/sgg_vis.jpg" width="300">
    <img src="figs/sgg_vis2.jpg" width="500">
</div>
<div align="left">
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    Left/Right: Biased/Debiased Scene Graphs.
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    Top/Bottom: Biased/Debiased Scene Graphs.
</div>

## Models
The source code will be available at this repository soon. Stay tuned (by clicking "watch" this repo) :)

## Citation
Coming soon.

## Credits
Our codebase is based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).
