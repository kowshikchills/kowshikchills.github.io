---
layout: post
title:  Reinforcement learning for Covid- 19- Simulation and Optimal Policy
author: kowshik
categories: [Reinforcement Learning]
image: assets/images/B1-1.jpg
tags: ReinforcementLearning tutorial
---

While the ML community is wondering how they could help the war against the COVID-19 pandemic, I decided to use reinforcement learning to tackle this crisis. This investigation yielded some interesting results in finding the set of optimal actions to fight virus spread.

## 1. Introduction

Imagine you are playing a **pandemic control game.** Your objective is to control the spreading of the virus with the least economic disruption. You can choose between a multitude of actions like ‘close all infected residential areas’, ‘run tests in infected areas’, ‘lockdown’ etc.

But the immediate question is: how do I quantify economic disruption? Fairly, we can assume that wider the restriction on the movement of the people, the worse the economic health. So our objective is to control the virus spread with the least impediment on the movement of the population.

What if an algorithm gives you a trained agent that can take actions on your behalf to achieve the goals you set? Would you not employ such an intelligent agent to curb the virus spread? The subject of reinforcement learning(RL) is around modeling such an intelligent agent.
> # The most exciting part of this modelling is that we can design an agent that curbs the virus spread in the long term with the least disruption to the economic activity.

## 2. Reinforcement Learning

Reinforcement Learning is a subfield of machine learning that teaches an agent how to choose an action from its action space. It interacts with an environment, in order to maximize rewards over time. Complex enough? let’s break this definition for better understanding.

**Agent**: The program you train, with the aim of doing a job you specify.<br/> 
**Environment**: The world in which the agent performs actions.<br/> 
**Action**: A move made by the agent, which *causes a change *in the environment.<br/> 
**Rewards**: The evaluation of an action, which is like feedback.<br/> 

In any RL modelling task, it’s imperative to define these **4 essential elements**. Before we define these elements for our Covid-19 problem, let’s first try to understand with an example: *how agent learn actions in an environment?*


**Agent**: Program controlling the movement of limbs<br/>  
**Environment**: The real world simulating gravity and laws of motion<br/>  
**Action**: Move limb L with Θ degrees<br/>  
**Reward**: Positive when it approaches destination; negative when it falls down.<br/> 

![walking]({{ site.baseurl }}/assets/images/B1-2.jpg)



Agents learn in an interactive environment by **trial and error** using feedback (Reward) from its own actions and experiences. Agent essentially tries different actions on the environment and learns from the feedback that it gets back. The goal is to find a suitable action policy that would maximize the **total cumulative reward** of the agent.

## 3. Pandemic Control Problem

Now let’s define these 4 essential elements for our pandemic control problem:
**Agent**: A Program controlling the movement of the citizens through different actions.<br/> 
**Environment**: The virtual city where the virus is spreading. By restricting the citizen’s movement, spread dynamics can be altered.<br/> 
**Action**: Control the movement of the citizens.<br/>  
**Rewards**: minimise infected from virus spread (pandemic control) +minimise people quarantined( least economic disruption)+ minimise people dead<br/> 

Now we need to code-up and discuss each element of this optimal control problem. let’s start with pandemic simulation environment.

## 4. Pandemic Simulation Environment
> # Model the whole pandemic transmission dynamics as interactions between different components.

Though there are a large number of pandemic simulation models, I decided to use my own simulation model drawing inspiration from the network model. I choose not to use the standard model because of the following reasons:

 1. In existing simulation models, the transmission dynamics of the virus does not react to the actions taken by the decision maker/agent. (eg. How would closing public transport impact virus spreading).<br/> 

 2. Existing transmission models doesn’t output a comprehensive observation on the state of the city.<br/> 

In order to prepare such an environment that overcomes above-mentioned shortcomings, I decided to break the whole pandemic transmission dynamics into interactions between different **components**.<br/> 

Let’s discuss these components and their respective assumptions of pandemic simulation environment. We will classify these components into Demographic Components, Transmission Dynamics, Contagious Components.

### Demographic Components

These are basic components of the simulation model on which the whole transmission dynamics are built. We will create a closed city where we intend to simulate the virus spread. There are assumptions considered about this city, such that the simulation process is less computationally expensive and also close to reality.

![walking]({{ site.baseurl }}/assets/images/B1-3.jpg)

### Transmission Dynamics

These transmission dynamics decide the extent and intensity of the virus spread. We can simulate any pandemic using these transmission dynamics.

![walking]({{ site.baseurl }}/assets/images/B1-4.gif)

As you can clearly visualize: Infected citizen makes the daily trip and he/she infects other citizens who came in **contact** with him with the **probability of transmission** at each unit. 
We essentially need to define how many citizens come in contact with the infected and what is the probability of transmission at each unit.

![walking]({{ site.baseurl }}/assets/images/B1-5.jpg)

### Contagious Components and Simulation Results

These contagious components help us build an environment. For a decision maker to take actions to curb the virus spread, he must understand the state of the infected city*( eg. number of citizens infected, number of residential areas infected, number of citizens quarantined ,etc).* 
These components facilitate the logging of infected/interaction information in a structured manner. We use the compartment model for simulation. 
Let’s simulate a simple compartment model with infinite hospital capacity. We will randomly infect 3 citizens and simulate a pandemic following the above transmission dynamics.

![walking]({{ site.baseurl }}/assets/images/B1-6.gif)

**Contagious Compartment:** All those active citizens who are infected and contagious are included in this list<br/> 
**Recognized Compartment:** All those infected who came to the governments notice.<br/>  
**Hospitalized Compartment**: All those infected citizens recognized by the government will be put in the hospital. Once the infected citizen enter this list, he will be removed from the Contagious Compartment.<br/> 
**Hospital Infrastructure Capacity**: The capacity of the hospital is limited. Once the capacity reaches, further infected citizens cannot enter the Hospitalized Compartment. This is a very important variable in our simulation, which you will see in plot 6.<br/>  
**Death**: Infected will be dead as the days progress with the probability proportional to his age<br/>

Let’s look at the simulation results for the pandemic in a city of 1L population and with infinite hospital Infrastructure Capacity and limited(500) capacity. Also, we need to compare it with standard epidemiological models.<br/>

![walking]({{ site.baseurl }}/assets/images/B1-7.jpg)

![walking]({{ site.baseurl }}/assets/images/B1-8.jpg)

>  This is a simple epidemiological model. The “ contagious line” in my simulation model(**Plot 6**) is closer to the “infected line” in SIR model(Plot 7). This clearly implies that the pandemic simulation is accurate.

## 5. Actions

The need for creating a new environment for the pandemic problem is essentially because we ideally want our pandemic simulation environment to react to the actions taken by the decision maker. So defining action space is as important as defining the environment. 
So by defining wide action space, we are enriching the decision maker’s choices to curb the virus spread.

The virus spread can be effectively curbed by: 
1. Restricting the movement of the citizens <br/>
2. Conducting the tests on probable citizens, so that infected citizens come to the government’s note before the symptoms kick-in.<br/>

You will now clearly see why I introduced the concept of transmission dynamics. By restricting the movement of the citizens, they are not susceptible to infection anymore. This condition can be easily embedded into the simulation and the dynamics of the virus spread change accordingly.

![walking]({{ site.baseurl }}/assets/images/B1-9.jpg)

These are the actions defined for the decision maker. 
For example, if the decision maker chooses action: 8 (**lockdown**): then all the citizens in the city cannot move.

The idea behind defining this action space is that we want to find the most **optimal action policy** of restricting citizen’s movement. We can design more actions, but for now, we limit to this action space.

## 6. Agent and Rewards

Out of 4 essential elements of Reinforcement Learning, we discussed *1. Environment 2. Actions* for our pandemic control problem. Let’s discuss agent and reward in this section.

An **agent** is essentially a program you train, with the aim of doing a job you specify. *But how do we specify the job*? *How can an agent understand your(decision maker) objectives? *The answer is through **reward**. The agent always try to find out the action policy that maximize the cumulative sum of rewards. So if we can tie the goals of the pandemic control problem with the reward function, we can train an agent which achieves goals for us.

Let’s reiterate our **objective**: To control the virus spread with the least impediment on the movement of the population( least economic disruption). 
So we need to minimize:<br/>
1. Number of people Infected (𝜨𝒊)<br/>
2. Number of people quarantined(𝜨𝒒)<br/>
3. Number of people died because of infection(𝜨𝒅)<br/>
We don’t essentially give equal weights to each number. For example, governments don’t let the economy remain healthy at the cost of citizens.<br/>

![walking]({{ site.baseurl }}/assets/images/B1-10.jpg)

One thing must be kept in mind when deciding 𝑤𝒊, 𝑤𝒒, 𝑤𝒅. Apart from their ethical importance, these weights are just numbers. We need to choose them judicially such that the agent actually learns to achieve the objectives we set.<br/>

In section 2( RL), we learnt how agent trains. Let’s try to understand the training process in the pandemic control problem. I used the DQN model to train the agent. In this DQN model, the agent tries random actions in the beginning (exploratory) to learn optimal action policy. An interesting concept in this model is **discounted sum of rewards**: agent gives lesser importance to the immediate rewards and strives to achieve long terms goals.<br/>

I will briefly explain this RL model: Q-learning learns the action-value function *Q(s, a)*: **how good to take an action at a particular observation**.<br/>

Let’s try to understand Q value: Consider the pandemic simulation environment, for a given observation:<br/>
*{infected, hospitalized, dead, exposed, infected houses, average age of infected}*<br/>

Agent will learns Q value **(expected rewards)** for each action ( Total 16 actions). The agent chooses the action with the highest Q value. We will limit the discussion on RL modelling techniques and jump into the results and Interpretation.

## 7. Results and Interpretation

Now we reach the end and also the most interesting part of this blog.

So let’s create a pandemic simulation in a city of size 1 Lakh. We will let the DQN agent take actions from its action space A *(plot 8)* to maximize the reward R*( Equation 1).*

![walking]({{ site.baseurl }}/assets/images/B1-11.jpg)

![walking]({{ site.baseurl }}/assets/images/B1-12.jpg)

![walking]({{ site.baseurl }}/assets/images/B1-13.jpg)

![walking]({{ site.baseurl }}/assets/images/B1-14.jpg)

## 8. Summary

This modelling and simulation can be extended to cities of different sizes. The actions taken by the agent are more intuitive as the agent understands/learns the pandemic simulation environment better. For example, agents choose to do a lot of tests in infected areas at the beginning of the spread. More action spaces and better reward function makes this whole RL modelling even closer to reality.

As I mentioned in the beginning, the intention behind writing this blog is to explore the possibility of collaboration and help the war against the corona virus spread. If anyone believes that they can contribute to this RL project, please feel free to mail me kowshikchilamkurthy@gmail.com. Also, I would love to take suggestions from you for better simulation and better RL modelling.

*references:*
1.[https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model)
2. [https://blogs.mathworks.com/headlines/2019/05/16/robot-quickly-teaches-itself-to-walk-using-reinforcement-learning/](https://blogs.mathworks.com/headlines/2019/05/16/robot-quickly-teaches-itself-to-walk-using-reinforcement-learning/)
3. H. S. Rodrigues, M. T. T. Monteiro, and D. F. M. Torres, “Dynamics of dengue epidemics when using optimal control,” *Mathematical and Computer Modelling*, vol. 52, no. 9–10, pp. 1667–1673, 2010.
