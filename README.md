# <p align=center> Find True Collaborators: Banzhaf Index-based Cross View Alignment for Partially View-aligned Clustering </p>

> **Auhtors:**
Shanghui Deng, Xiao Zheng, Chang Tang, Kun Sun, Yuanyuan Liu, Xinwang Liu

This repository contains simple pytorch implementation of our paper [BIN](https://dl.acm.org/doi/abs/10.1145/3746027.3754826)

### 1. Overview

<p align="center">
  <img src="assest/BIN.png"/><br/>
</p>

Framework overview of BIN. We first encode the misaligned multi-view data into view-specific latent spaces. Next, we model the samples as game players and optimize collaboration using three loss terms: Banzhaf gain loss $$\mathcal{L}_{\mathfrak{B}}$$, contrast loss $$\mathcal{L}_C$$, and reconstruction loss $$\mathcal{L}_{rec}$$. Based on this, we solve the optimal matching problem to effectively pair the samples, enabling cross-view fusion for the final clustering.


