---
layout: post
title:  Neural Hawkes Process
author: kowshik
categories: [Point Processes]
image: assets/images/B4-1.jpg
tags: [featured]
---


## Neural Hawkes Process

In the last blog in this publication, we looked at the definition of point process and also looked at Hawkes process in detail. In this blog, we will understand how to construct a neurally self-modulating multivariate point process in which the conditional intensity functions of multiple events is modelled according to a LSTM. It is assumed that reader is aware of some basic concepts of LSTM like hidden state, memory cell and memory control gates.

## Introduction

In the vanilla Hawkes process, the conditional intensity function is constructed using background intensity and excitation function. Past events can temporarily raise the probability of future events. However, in the real world application, the flexibility of vanilla Hawkes process is very limited to approximate many sections of the problems. We need a better model with least assumptions to approximate the conditional intensity function. We can generalise the Hawkes process by determining the event intensities using a neural network architecture.

We will try to understand why we need a neural network model to approximate a conditional intensity function. In the vanilla model, for example lets consider exponential decay Hawkes process ( [Understanding point Process](https://medium.com/point-processes/understanding-point-processes-6e3d2f6c5480)).

![walking]({{ site.baseurl }}/assets/images/B4-2.jpg)

We assume that each arrival of event excites the future events positive and additive in nature and exponentially decaying with time. Parameters α, β can be interpreted as that each arrival in the system instantaneously increases the arrival intensity by α, then over time this arrival’s influence decays at rate β. Parameter α fails to capture inhibition effects, this limits the expressivity of the model.

## Neural Network Approximation

In many real-world patterns doesn’t follow these assumptions. The effect of past events on future events can now be additive even subtractive, and can depend on the sequential ordering of the past events. For example, its not really simple to accurately model the earthquakes and complex after-shocks by just assuming exponentially decaying conditional intensity function. We can further generalise the Hawkes process by using LSTM to model the conditional intensity functions from the **hidden state** of a recurrent neural network. Long short-term memory (LSTM) is a modified recurrent neural network (RNN), built to handle the vanishing gradients problem RNN faces. It process entire sequences of data instead of just single data points. The hidden state is a deterministic function of the past history.

Neural Hakes process removes the restriction that the past events have independent, additive influence on λ٭(t). This method uses a recurrent neural network to predict λ٭(t). This allows learning underlying dynamics and approximate the conditional intensity function based on history H(t): number, order, and timing of past events. Let assume that we are solving **multi-variate or mutually exciting point processes,** these are essentially set of one-dimensional point processes which excite themselves and each other.

Event type k has a time-varying intensity λk٭(t) is approximate by a value in hidden state vector h(t). In LSTM, h(t) is a sufficient statistic of the history and approximately summarise the past event sequence. The Neural Hawkes process is different from exponential decay Hawkes process in following ways:

 1. The base rate is not a constant λ, but can change with time.

 2. The excitation can be can be non-monotonic, because the influences on λk٭(t) can be excitatory and inhibitory.

 3. The model can generalise the influence of events on other events, also takes into account that some pairs of event types do *not *influence one another.

If we closely observe, the familiar discrete-time LSTM is developed on discrete time steps, but it cannot directly consume the event sequence for generating the conditional intensity function thorough its hidden states. 
Thus we are using continuous-time LSTM, which is familiar to discrete-time LSTM. The difference is that in the continuous interval following an event, each memory cell c exponentially decays at some rate δ toward some steady-state value[1].

![walking]({{ site.baseurl }}/assets/images/B4-3.jpg)


We can use this loss function for optimising the LSTM parameters. The derivation of this equation is not discussed in the blog, readers who are interested to understand the mathematical rigour can appendix B.1 in the paper in references[1].

## Summary

We discussed a more flexible multivariate Hawkes process using the novel continuous-time recurrent neural network (LSTM). This significantly increased the expressibility of the model. This model can be used in cases where set of events can have both inhibition and excitation effect based on the sequence of the events. This model has interesting applications in various business problems like predictive maintenance, advetising etc. In the next blog in this series we will discuss one of its applications in solving real world problems.

Thanks for your time :)

### References

[1] The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process: [https://arxiv.org/abs/1612.09328](https://arxiv.org/abs/1612.09328)
