---
layout: post
title:  "Strategies for Customer Retention: A Cox Survival Model Treatment"
author: dharani
categories: [Cox Process]
image: assets/images/TP1-1.jpg
tags: CoxProcess
---
#### Techniques to devise personalized strategies using statistical models

![walking]({{ site.baseurl }}/assets/images/TP1-2.jpg)

## Introduction

Customer churn occurs when customers or subscribers discontinue their association with a company or service. There are many Machine Learning models to predict if a customer is going to churn or not. The problem doesn’t stop there, business has to deploy certain strategies to retain the customers who are at the verge of churn because it’s [five times cheaper](https://www.forbes.com/sites/jiawertz/2018/09/12/dont-spend-5-times-more-attracting-new-customers-nurture-the-existing-ones/#4b86da1f5a8e) to retain an existing customer than to acquire a new one. Statistical models can be used to derive and evaluate personalized strategies which is a core challenge in CPG companies.

We call the event of customer churn as ***failure*** and ***survival time*** is the time taken for such failure/churn. Survival models are statistical techniques used to estimate the time span taken for an event to occur. Cox Proportional-Hazards model is a popular statistical model for survival analysis. Using churn data set from Kaggle, we will try to use this model to understand customer behavior and compare different strategies that can improve customer retention.

Please refer to [this blog](https://medium.com/point-processes/the-cox-proportional-hazards-model-da61616e2e50) to understand Mathematical Equations and reason behind using this.

## Road map to enhance customer retention rate:

 1. Customer’s characteristics and demographics play a pivotal role in understanding retention behavior. Our goal is to understand the relation between these features and survival time(time taken to churn). We can plot survival/retention curves that are specific to a customer to gain valuable insights.

 2. Devise personalized strategies(for example, increase incentives/offers)for high-valued customers for different survival risk segments during the time. Our goal is to evaluate and compare how they improve the survival/retention behavior in a customer.

## A Quick Recap of Cox Proportional-Hazards Model

*Cox proportional-hazards model* is developed by Cox and published in his work[1] in 1972.The most interesting aspect of this survival modeling is its ability to examine the relationship between survival time and predictors. For example, if we are examining the survival of patients then the predictors can be age, blood pressure, gender, smoking habits, etc. These predictors are usually termed as covariates.

![walking]({{ site.baseurl }}/assets/images/TP1-3.jpg)

 1. *H*azard function **λ(t)**: gives the instantaneous risk of demise at time t

 2. Z: Vector of features/covariates

 3. ***λo(t)*** is called the baseline hazard function: Describes how the risk of event changes over time. It is underlying hazard with all covariates equal to 0.

## Model Implementation on churn data set:

### Problem Setup & Data Engineering

I have taken telecom customer churn data set. Lets check the data structure:

    import pandas as pd
    df = pd.read_csv(‘Data_Churn_Telecom_Cox.csv’)
    df.head()

![walking]({{ site.baseurl }}/assets/images/TP1-4.jpg)

![walking]({{ site.baseurl }}/assets/images/TP1-5.jpg)


>  These **features** gives the customer’s **demographics** and **characteristics** / **behaviour**. There are 96 such features.

“**Total number of months in service**” column gives us the survival/retention time of a customer. “**churn**” column gives whether customer churns or not i.e., event occurrence.

### Data Engineering

Listed down are the *feature engineering steps* and we also look at the distributions for some of these features. Code for each feature engineering step is published at end of the blog.

![walking]({{ site.baseurl }}/assets/images/TP1-6.jpg)

A fairly simple assumption is **proportional hazards**, which is crucial in Cox regression that is included in its name(the Cox proportional hazards model). It means that the *ratio* of the hazards for any two individuals is constant over time. We drop those features if they don’t pass this condition.

## Survival Analysis

Here comes the most interesting section: the implementation of cox model in python with the help of lifelines package. Understanding the impact of features on survival rates helps us in predicting the retention of a customer profile. The Cox model assumes that each feature have an impact on the survival/retention rate.

One of the basic assumptions of the CPH model is that the features are not collinear. We can either solve the issue of multi-collinearity before fitting the Cox model or we can apply a penalty to the size of the coefficients during regression. Using a penalty improves stability of the estimates and controls high correlation between covariates.

    from lifelines import CoxPHFittr
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df_f, duration_col=’Total number of months in service’, event_col=’churn’)
    cph.summary #output2
    cph.plot   #output3

![walking]({{ site.baseurl }}/assets/images/TP1-7.jpg)

## Interpreting the summary

* Hazard ratio (HR) given by exp(coef), where coef is the weight corresponding to the feature. If exp(coef) = 1 for a feature, then it has no effect. If exp(coef) > 1, decreases the hazard: **improves the survival**.

* *number of unique subscribers in the household* has HR = 1.35 which improves survival/retention. *mean number of unanswered data calls* has HR = 0.16 implies it has bad effect on survival rate.

## Results & Visualization

The best way to understand impact of each features/decision is to plot the survival curves for single feature/decision by keeping all other customers characteristics/demographics unchanged. we use [**plot_covariate_groups()](https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html#lifelines.fitters.coxph_fitter.CoxPHFitter.plot_covariate_groups)** method and pass the arguments — feature of interest and the values to display.

### Survival Profiles of Features

One quick way to interpret these different survival curves is that the decision with corresponding survival curve leaning to the right yields more survival/retention probability than that to its left. A plot below for Average monthly revenue over the life of the customer =400 has more survival probability as it is to the right compared to that of 10 which is to its left.

![walking]({{ site.baseurl }}/assets/images/TP1-8.jpg)

### Personalized strategies for Customers

We can plot the survival profiles for each customer and analyse the reasons for low survival/retention rates by looking at customer features. From above discussions, we already know what actions can improve the survival rates of the customer.

![walking]({{ site.baseurl }}/assets/images/TP1-9.jpg)

We can plot the survival profiles of each customer. For time being let’s consider customer with ID 1032424 and compare the two strategies as shown below:

![walking]({{ site.baseurl }}/assets/images/TP1-10.jpg)

We can clearly see that strategy 1( i*ssue more models to the customer : 1032424*) has comparatively longer survival time than strategy 2(Reduce revenue generated from the customer by giving offer). Similarly, we can analyse each and every customer and design proactive strategies to ensure highest retention statistically. We can also compare different strategies developed by business intelligence teams and deploy based on the effectiveness of a strategy to retain customers.

## Summary

We segmented customer behavior by grouping them based on average monthly revenue brackets, number of models issued etc., and Cox proportional-hazards model enabled us to derive personalized strategies to reduce the churn rate. Not only deriving personalized strategies, we learnt to compare them. As an example, for one customer we saw in above section, statistically proves that issuing more models will have a better impact than providing incentives/offers for a customer to stick to the company for a longer time span. This is an outstanding way to meet the landscape of customer expectations and increase customer engagement with the company.

Thanks for your time :)

Full code for reference:


    import pandas as pd
    df = pd.read_csv('Data_Churn_Telecom_Cox.csv')
    with open('cols.txt') as f:
       lines = f.readlines()
    cols_names = {}
    for i in range(len(lines)):
       for j in df.columns:
           if j == lines[i][:-1]:
               cols_names[j] = lines[i+1][:-1]
    cols_names['Customer_ID'] = 'Customer_ID'
    df.columns =   cols_names.values()


    df['churn'] = df['Instance of churn between 31-60 days after observation date']
    del  df['Instance of churn between 31-60 days after observation date']
    df = df.dropna()

    del df['N']
    df.set_index('Customer_ID', inplace=True)
    '''
    Drop categorical features with unique values >2
    '''
    import numpy as np
    df_str = df.loc[:, df.dtypes == object]
    for i in df_str.columns:
       if len(np.unique(df_str[i].values)) >2:
           del df[i]
    '''
    One hot encoding
    '''
    df_str = df.loc[:, df.dtypes == object]
    for i in df_str.columns:
       one_hot = pd.get_dummies(df[i])
       one_hot.columns = [i +'_'+j for j in one_hot.columns]
       df = df.drop(i,axis = 1)
       df = df.join(one_hot)

    survival_time = df['Total number of months in service'].values
    del df['Total number of months in service']
    churn = df['churn'].values
    del df['churn']


    '''
    Drop correlated features
    '''

    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]
    df.drop(to_drop, axis=1, inplace=True)



    df = df[list(df.columns[:69])+['Credit card indicator_N']]
    df['Total number of months in service'] = survival_time
    df['churn'] = churn
    df = df[df['churn'] == 1]

    '''
    Select valuable features
    '''
    df_sampled = df.sample(n=1000)
    from lifelines import CoxPHFitter
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(df_sampled, duration_col='Total number of months in service', event_col='churn')
    df_stats = cph.summary
    features_valuable = list(df_stats[df_stats['exp(coef)'].values > 1.01].index) + list(df_stats[df_stats['exp(coef)'].values < 0.98].index)
    df = df[features_valuable+['churn','Total number of months in service']]



    from lifelines import CoxPHFitter
    cph = CoxPHFitter(penalizer=0.01)
