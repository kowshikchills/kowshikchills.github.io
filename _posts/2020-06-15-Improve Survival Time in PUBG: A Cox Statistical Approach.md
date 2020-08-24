---
layout: post
title:  Improve Survival Time in PUBG- A Cox Statistical Approach
author: kowshik
categories: [Cox Process]
image: assets/images/B7-2.jpg
tags: CoxProcess
---

### A Real World Application of Cox Proportional-Hazards Model


## Introduction

PUBG needs no introduction. It is one of the popular and the most played games right now. Players fight to death until one remains, so it is a survival game. There are pure statistical models to analyse the survival times. Using **PUBG data, **we will try to use one such survival models to understand how different strategies can improve the player’s survival rates.

This blog is written for tech, non-tech readers and most importantly PUBG players. I will also include my python implementation for the benefit of tech readers. This can be seen as a sequel to my blog: [*The Cox Proportional-Hazards Model](https://medium.com/point-processes/the-cox-proportional-hazards-model-da61616e2e50) and must read for those who are interested in understanding the mathematical background and python implementation of magical *Cox Proportional-Hazards Model.*

We will use the data published in Kaggle [datasets ](https://www.kaggle.com/skihikingkevin/pubg-match-deaths?select=aggregate)where there are over 720,000 PUBG matches. The data log was extracted from [pubg.op.gg](http://pubg.op.gg/), a game tracker website. We will use this data log to understand different modes of game strategies using statistical models and try to figure out the method to evaluate the strategies.

### A Quick Recap of Cox Proportional-Hazards Model

*Cox proportional-hazards model* is developed by Cox and published in his work[1] in 1972. It is the most commonly used regression model for survival data. The most interesting aspect of this survival modeling is it’s ability to examine the relationship between survival time and predictors. For example, if we are examining the survival of patients then the predictors can be age, blood pressure, gender, smoking habits, etc. These predictors are usually termed as covariates. *Note: It must not be confused with linear regression, the assumptions might be linear in both regression and survival analysis but the underlying concepts are different. Methods we employ for parameter estimations of regression model and survival model are very different from each other.*

![centre]({{ site.baseurl }}/assets/images/B7-3.jpg)

 1. *H*azard function **λ(t)**: gives the instantaneous risk of demise at time t

 2. Z: Vector of features/covariates

 3. ***λo(t)*** is called the baseline hazard function

## PUBG Problem Setup & Data Engineering

Let’s have a look at the raw data before we define the problem setup.

    import pandas as pd
    df = pd.read_csv(‘agg_match_stats_0.csv’)
    df.head()

![centre]({{ site.baseurl }}/assets/images/B7-4.jpg)

*Feature Description***: player_size**: Team Size, **player_dist_ride**: Distance covered using vehicle by the player , **player_dist_walk**: Distance walked by the player, **player_kills**: Number of kills by the player, **players_survive_time**: time survived by the player

### Problem Setup

Players in PUBG can choose different strategies to maximise the survival time. We define strategy as a combination of one or more *player’s decisions*. Strategies can be something like:

1. Travel extensively with least confrontation with enemies,
2. Use a motorised vehicle most of the time,
3. Only Walk, but confront with enemies more often, or 
4. Even something funnier like: Play only afternoons over the weekends😝.

There can be 1000’s of such strategies, some of them might look trivial other might not. Our goal is to find a way to evaluate these strategies based on their survival rates. Apart from raw data provided, we also need to engineer these columns to derive meaningful features(player decisions).

### Data Engineering

In this section, we will briefly discuss the features needed to be extract from the raw data available to use. These features can be simply seen as the decisions taken by the player. Let’s list them and also look at the distributions for some of these features.

![centre]({{ site.baseurl }}/assets/images/B7-5.jpg)

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy import stats
    df = pd.read_csv('agg_match_stats_0.csv')
    df_features = create_features(df) #func is defined at end of blog
    df_features.head()

![centre]({{ site.baseurl }}/assets/images/B7-6.jpg)

Now that we extracted the features, lets jump into the implementation of cox proportional-hazards model.

## Survival Analysis

This is the most interesting section: the implementation of cox model in python with the help of lifelines package. It is very important to know about the impact of features on the survival rates. This would help us in predicting the survival rates of a PUBG player, if we know the associated feature values. The Cox model assumes that each features have an impact on the survival rates.

One of the basic assumptions of the CPH model is that the features are not collinear. We can either solve the issue of multi-collinearity before fitting the cox model or we can apply a penalty to the size of the coefficients during regression. Using a penalty improves stability of the estimates and controls for high correlation between covariates.

    from lifelines import CoxPHFitter
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df, duration_col='player_survive_time', event_col='dead')
    cph.plot()

![centre]({{ site.baseurl }}/assets/images/B7-7.jpg)

Coefficients of the features which indicate the measure of the impact on the survival rates of the PUBG player.

### Interpreting the summary

* Hazard ratio (HR) given by exp(coef), where coef is the weight corresposing to the feature. If exp(coef) = 1 for a feature, then it has no effect. If exp(coef) > 1, decreases the hazard: **improves the survival**.

* *weekend_indi*( that is whether player player over weekend or weekday ) doesn’t play any significant role in predicting his survival risk, whereas *player_kills* ( number of kills by player) variable plays significant role in predicting survival risk .

* *game size *feature with exp(coef) = 1.0 has **no effect** on the survival rates: so it implies that the survival of the player does not depend on the *game size.*

* *%player_dist_ride *feature with exp(coef) = 1.73 (>1) this is good for survival. So preferring the **vehicle** instead of **walking** increases the survival rates.

For better understanding of the math behind above deductions, please refer to my earlier blog: [*The Cox Proportional-Hazards Model](https://medium.com/point-processes/the-cox-proportional-hazards-model-da61616e2e50). 
*In the next section, we will also see how different features play together to decide the survival rates of the PUBG player.

## Results & Visualisation

The best way we understand impact of each features/decision is that we plot the survival curves for single feature/decision i.e., we keep all other player’s decisions unchanged. we use[**plot_covariate_groups()](https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html#lifelines.fitters.coxph_fitter.CoxPHFitter.plot_covariate_groups)** method and give it the feature of interest, and the values to display. Also we will look at the survival rates for different strategies ( combination of decisions)

In this section we will discuss 

1. Survival profiles of Decisions
2. Survival profiles of Strategies

### Survival profiles of Decisions

One quick way to interpret these different survival curves is that the decision with corresponding survival curve leaning to the right yields more survival probability than that of its left. Let’s try to understand this with an example.

![centre]({{ site.baseurl }}/assets/images/B7-8.jpg)

![centre]({{ site.baseurl }}/assets/images/B7-9.jpg)

![centre]({{ site.baseurl }}/assets/images/B7-10.jpg)

**Interpreting plot 3**

* It clearly implies that the survival time of PUBG player increases if he choose to walk instead of taking a vehicle

* More the distance he traverses, better his survival rates (which is intuitive)

### Survival profiles of Strategies

Let’s quickly see the survival profile for different strategies. For example, consider these four strategies:
1. Use vehicles extensively, travel longer distances and kill often
2. Only walk, travel smaller distances and don’t confront with enemies often 
3. Do team work, use vehicle less often and travel large distances
4. Select a match with small number of players and kill extensively

![centre]({{ site.baseurl }}/assets/images/B7-11.jpg)

The values for decisions are fixed as per the above 4 strategies

![centre]({{ site.baseurl }}/assets/images/B7-12.jpg)

Even in the real world survival situations, moving and confronting with the enemies is better than staying idle. We can handcraft 1000’s of such strategies and compare their survival behaviours. We can even understand and approximate the human behaviour during survival situations by applying these kind of statistical model on the data extracted from the survival games.

## Summary

We looked at a real world application of Cox proportional-hazards model. We understood how different strategies impact the survival times of the PUBG player. Out of those strategies we analysed, we found strategy of “*using vehicles extensively, travelling longer distances and killing often*” statistically promising the longest survival of a PUBG player in a match. There are also neural network variants of Cox proportional-hazards model, we will look at such neural variant of Cox PH model in my next blog in this series.

Thanks for your time :)

Here is the full code for reference:


#import all the dependencies

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

#import data set
df = pd.read_csv('agg_match_stats_0.csv')

#Create new features

    def create_features(df)
        df['player_survive_time'] = df['player_survive_time']/60
        df['date'] = [i.split('+')[0] for i in df['date'].values]
        df['date'] = pd.to_datetime(df['date'],format='%Y-%m-%dT%H:%M:%S')
        df['dayofweek_num']=df['date'].dt.dayofweek  
        df['dayofweek_name']=df['date'].dt.weekday_name
        df['Hour'] = df['date'].dt.hour 
        df['weekend_indi'] = 0       
        df.loc[df['dayofweek_num'].isin([5, 6]), 'weekend_indi'] = 1
        df['time_of_day'] = 0       
        df.loc[df['Hour'].isin([24,1,2,3,4,5,6]), 'time_of_day'] = "LateNight"
        df.loc[df['Hour'].isin([7,8,9,10,11]), 'time_of_day'] = "Morn"
        df.loc[df['Hour'].isin([12,13,14,15,16,17,18,19]), 'time_of_day'] = "Evening"
        df.loc[df['Hour'].isin([20,21,22,23]), 'time_of_day'] = "Night"
        df['%player_dist_ride'] = df['player_dist_ride']/(df['player_dist_ride']+df['player_dist_walk'])
        df['%player_dist_walk'] = df['player_dist_walk']/(df['player_dist_ride']+df['player_dist_walk'])
        df['total distance'] = df['player_dist_ride']+df['player_dist_walk']
        df['only_walk'] = 0       
        df.loc[df['%player_dist_walk'].isin([1]), 'only_walk'] = 1
        for i in ['date','team_id','team_placement','player_name','player_dmg','player_dbno','player_dist_ride',
                 'player_dist_walk','Hour','dayofweek_num','dayofweek_name','match_id','match_mode']:
            del df[i]
        one_hot = pd.get_dummies(df['time_of_day'])
        df = df.drop('time_of_day',axis = 1)
        df = df.join(one_hot)
        del df[0]
        df['dead'] = 1
        df = df.dropna()
        return(df)


    df_sampled = create_features(df)

    from lifelines import CoxPHFitter
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df_sampled, duration_col='player_survive_time', event_col='dead')


    '''

    Individual features 
    '''
    cph.plot_covariate_groups('only_walk', [0,1], cmap='coolwarm')
    plt.xlabel('time (minutes)')
    plt.ylabel('Survival Curve')

    '''
    strategies

    '''
    df_strategy = pd.DataFrame()
    df_strategy['game_size'] = [60,60,60,30]
    df_strategy['party_size'] = [2,2,2,1]
    df_strategy['player_assists'] = [1,0,4,1]
    df_strategy['player_kills'] = [6,1,2,5]
    df_strategy['weekend_indi'] = [0,0,0,0,]
    df_strategy['%player_dist_ride'] = [0.8,0,0.2,0.5]
    df_strategy['%player_dist_walk'] = [0.2,1,0.8,0.5]
    df_strategy['total distance'] = [9000,3000,7000,4000]
    df_strategy['only_walk'] = [0,1,0,0]
    df_strategy['Evening'] = [1,0,1,1]
    df_strategy['LateNight'] = [0,1,0,1]
    df_strategy['Morn'] = [0,0,1,0]
    df_strategy['Night'] = [0,0,0,0]
    df_strategy.index = ['Strategy 1','Strategy 2','Strategy 3','Strategy 4']

    cph.predict_survival_function(df_strategy).plot()
    plt.xlabel('time (minutes)')
    plt.ylabel('Survival Curve')

