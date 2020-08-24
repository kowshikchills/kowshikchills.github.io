---
layout: post
title:  The Cox Proportional-Hazards Model
author: kowshik
categories: [Cox Process]
image: assets/images/B6-2.jpg
tags: CoxProcess
---

### A Modelling Technique to Estimate the Survival Rates


Modelling time has been a topic of interest for scientists, sociologists, and even epidemiologists. A maintenance engineer wants to predict the time it takes for the next failure of a particular component in a vehicle engine occurs so that he can schedule preventive maintenance. It is of epidemiologist’s interest to predict when the next outbreak will occur, so he can plan for medical interventions. Business analyst want to understand the time it takes for an high values customer to churn so that he/she can take preventions measures.

In our earlier blogs on point process model, we explored statistical techniques that estimate the likeliness of a certain event occurrence in the backdrop of the time dimension. In this new statistical techniques, we will keep the event in backdrop and model time. Survival models are statistical techniques used to estimate the length of time taken for an event to occur. We call event occurrence as *failure* and *survival time* is the time taken for such failure.

*Cox proportional-hazards model* is developed by Cox and published in his work[1] in 1972. It is the most commonly used regression model for survival data. The most interesting aspect of this survival modeling is it ability to examine the relationship between survival time and predictors. For example, if we are examining the survival of patients then the predictors can be age, blood pressure, gender, smoking habits, etc. These predictors are usually termed as covariates.

## *Hazard Function ( *λ(t) )

The *hazard function *λ(t) is defined as the event rate at time *t*. Suppose that an item has survived for a time t, then λ(t) is the probability that it will not survive for an additional time *dt. H*azard function λ(t) gives the instantaneous risk of demise at time t, conditional on survival to that time and covariates.

![walking]({{ site.baseurl }}/assets/images/B6-3.jpg)



 1. Z is a vector of covariates

 2. ***λo(t)*** is called the baseline hazard function

*Baseline hazard function* describes how the risk of event per time unit changes over time. It is underlying hazard with all covariates Z1, …, Zp equal to 0.

![walking]({{ site.baseurl }}/assets/images/B6-4.jpg)



## Parameters Estimation

Cox proposed a partial likelihood for β without involving baseline hazard function ***λo(t)***. The parameters of the Cox model can still be estimated by the method of partial likelihood without specifying the baseline hazard. The likelihood of the event to be observed occurring for subject j at time Xj can be written as

![walking]({{ site.baseurl }}/assets/images/B6-5.jpg)



Lⱼ(β) is probability that individual j fails give that there one failure from risk set. Partial Probability **L**(β) = **∏(L**ⱼ(β)).

R(Xj) is called risk set, it denote the set of individuals who are “at risk” for failure at time t *[3]*.

This partial likelihood function can be maximised over *β* to produce maximum partial likelihood estimates of the model parameters[2]. For convenience we apply the log to the partial likelihood function: 
**log-partial likelihood( 𝓁(β))**:

![walking]({{ site.baseurl }}/assets/images/B6-6.jpg)

We differentiate log-partial likelihood( 𝓁(β)) and equate it to zero for calculating the β. The partial likelihood can be maximised using the [Newton-Raphson](https://en.wikipedia.org/wiki/Newton%27s_method) algorithm[2].

## Python Implementation

Let’s jump into the final and most interesting section: implementation of CoxPH model in python with the help of lifelines package. An example dataset we will use is the Rossi recidivism dataset.

    **from** **lifelines** **import** CoxPHFitter
    **from** **lifelines.datasets** **import** load_rossi
    rossi_dataset = load_rossi()

![walking]({{ site.baseurl }}/assets/images/B6-7.jpg)



 1. arrest column is the event occurred,

 2. The other columns represent predicates or covariates

 3. Week is the time scale

    cph = CoxPHFitter()
    cph.fit(rossi_dataset, duration_col='week', event_col='arrest')
    cph.print_summary()

![walking]({{ site.baseurl }}/assets/images/B6-8.jpg)


![walking]({{ site.baseurl }}/assets/images/B6-9.jpg)



cph.plot() outputs this pictorial representation of coefficient for each predictor. The summary statistics above indicates the significance of the covariates in predicting the re-arrest risk. Age doesn’t play any significant role in predicting the re-arrest, whereas marriage variable plays significant role in predicting time for re-arrest.

Lets look at a survival curve for one candidate with particular features(predicates/ covariates) using cph.predict_survival_function(df_vector).plot(). **Survival rates (S(t))** simply gives us the probability that event will not occur beyond time t.

![walking]({{ site.baseurl }}/assets/images/B6-10.jpg)



we can also plot what the survival curves for single covariate i.e we keep all other covariates unchanged. This is useful to understand the impact of a covariate. we use[**plot_covariate_groups()](https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html#lifelines.fitters.coxph_fitter.CoxPHFitter.plot_covariate_groups)** method and give it the covariate of interest, and the values to display[4].

![walking]({{ site.baseurl }}/assets/images/B6-11.jpg)



We can clearly see that the survival rates of married prisoner is higher than that of unmarried as married tends less to do crimes again as he got family to take care. We can simply deduce such similar and valuable insights from the above survival curves.

## Summary

We introduced the most famous survival model: Cox model; in this blog and understood its mathematical implementation. We also saw through its python implementation that the model has kept its promise of interpretability. There are more and robust model to discuss in survival model. We will discuss more examples and other famous survival models in the next blog in this series.

*Thanks for your time😀*
