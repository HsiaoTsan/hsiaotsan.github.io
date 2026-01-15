---
title: "XXPO: The Family of RL Policy Optimization for LLM Post-Training"
date: 2025-12-31T12:14:04-08:00
draft: false
tags: ["Reinforcement Learning", "Large Language Models", "Policy Optimization"]
categories: ["General"]
description: "Welcome to my personal blog where I share thoughts on AI, research, and more."
summary: "Introducing my new blog built with Hugo and the PaperMod theme."
---

## Introduction

What are PPO, GRPO, GSPO, etc? What are their relations? After reading this article, you'll understand:

- **Why**: The motivation for each variant of policy optimization.
- **Connection**: How each algorithm connect to other algorithms in the XXPO family.

## Policy Optimization

PPO (Proximal Policy Optimization)
- actor: also known as the policy network, it outputs the action distribution;
- critic: also known as value network, it outputs the state value;

GRPO:
- removed critic network in PPO, save memory and computation;

Dr.GRPO (GRPO Done Right):
- vanilla GRPO unintentionally encourages longer answers to dilute the negative reward, e.g., -1/1000 > -1/100.

GSPO:
- token-level optimization computes gradient based on each token, where many tokens can have zero gradients due to clipped importance weight;
- sequence-level optimization computes gradient based on a complete sequence, where the importance weight is the geometric mean of token-level importance weight, hence more stable.
- Suitable for training Mixture-of-Experts (MoE) models.

Decoupled PPO:
- 'Decoupled' means treating $\pi_{\text{old}}$ in trust region constraint and importance weight *differently*: use the latest behavioral policy (closest to the target policy $\pi_{\theta}$) as a proximal policy $\pi_{\text{prox}}$ for trust region constraint and keep using the actual behavioral policy of a sequence as $\pi_{\text{old}}$ for importance weight.
- specifically designed for async RL where the off-policyness is high: target policy $\pi_{\theta}$ can be several updates ahead of the behavior policy.

A-3PO (Approximated Proximal Policy Optimization):
- the computation of proximal policy $\pi_{\text{prox}}$ requires an extra forward pass in LLMs, which is expensive. A-3PO discovers that $\pi_{\text{prox}}$ only serves as an anchor to prevent excessive policy updates, therefore $\pi_{\text{prox}}$ can be interpolated between the behavior policy and the target policy $\pi_{\theta}$.

SPPO:

SAPO:
- instead of sudden clipping of gradient, do gradual decay on gradient when the importance weight deviates from 1.


