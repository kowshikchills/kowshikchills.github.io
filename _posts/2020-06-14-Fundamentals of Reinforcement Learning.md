---
layout: post
title:  Fundamentals of Reinforcement Learning
author: dharani
categories: [Reinforcement Learning]
image: assets/images/RL1-2.jpg
tags: RL
---
### *Learning decisions that makes the difference*


## Introduction

Designing machine that learn to do a job by itself is one of the most researched topic than any other in recent times due to various reasons like increased computational power, availability of resources to experiment etc., This lead to uncover significant innovations that made life simpler. If you just have data then an algorithm will provide insights or you train a model and it recognizes your face and many other use cases that we see around us which are built using Machine Learning and Deep Learning. Reinforcement Learning is burgeoning by gaining a lot of attention due to its proven capability in making **sequential** **decision** process.

## Concepts of RL

Reinforcement Learning basically consists of an **agent**(decision maker) that tries to learn from a **state** in a given surrounding that it interacts called **environment** and changes its state because of some **action** taken as per the feedback provided by the environment during the **episode**. This feedback is numerical(positive, negative or zero) and is called a **reward**. The optimal behavior of an agent is to learn such that it always gets good feedback i.e., maximize the reward by taking suitable actions. So in RL we are providing the scenario to the agent and it can figure out itself or discover how to take decisions in the most applaudable way.

![walking]({{ site.baseurl }}/assets/images/RL1-2.jpg)


Lets understand the terms used in RL using the very well known PUBG game as a simple example:

* The player in PUBG is an **agent** here and battleground is his **environment**

* A complete game played is an **episode** and walking, running etc are **states** — helps to pick actions

* The agent has number of **actions** to take like moving left, right, front and back, run, fire, kill, bend, jump, change gun etc.,

* The **reward** the agent gets here is positive if he kills and zero if he survives with the help of his teammates till the end or negative if he gets shot by other player.

In order to win the game we have to maximize the reward by taking suitable actions at each time frame. Simply we start from a state, take an action and change to another state and get a reward for that action and repeat the process to learn more about the environment setting.

There are few challenges RL has before us. Some of them are:

* Trade-offs: As we understood that agent has to optimize the rewards and also it has to interact continuously with the environment i.e., it needs to explore a lot. This leads to a trade off between exploration and exploitation. It has to choose whether it should keep exploring new states which might result in lower reward or take the path that has already seen and got quite a good set of rewards.

* Generalization: Can the agent understand or learn if the actions are good/bad in its previously unseen states.

* Delayed consequences: We also need to understand if an agent gives a high reward in a current state, it is because of just this state or a series of decisions that it has taken to reach this state?

There are few key concepts that are applicable in RL. A good knowledge on these will let us understand the formulation of agent’s decision process and model of the environment.


![walking]({{ site.baseurl }}/assets/images/RL1-3.jpg)

**Markov Property:** If an agent changes from one state to other it is called a transition and the probability at which it makes the transition is called transition probability. Generally if we have all the probabilities of an agent going from one state to the other, then it is represented in a transition matrix. Markov property says “*Future is independent of past given present*”. The equation below depicts the probability of transitioning to next state St+1 in time t is only dependent on the current state St and the action taken At and is independent of the history

![walking]({{ site.baseurl }}/assets/images/RL1-4.jpg)

This is the transition probability matrix which has all the transition probabilities of all states. For example, P12 depicts the probability of transitioning from state 1 to state 2 and so on.

When we traverse through a set of states in the environment which follows Markov Property, then it is called a Markov chain. They might include random set of states in the Markov chain that also have transition probabilities and we can compute the optimal chain that results in high reward.

* In RL, we are more concerned about optimizing the total reward that an agent receives from the environment rather than the immediate reward it gets by transitioning from one state to the other. So we measure the optimality using a function called **return(Gt)** which is sum of rewards the agent received from time t (Eq 1).

* In many games like Atari, alpha go, chess or PUBG we know the game is going to terminate after certain time steps. If this is the setting then it is called an **episodic** task. If we start another game then we are initiating a new episode so episodes are **independent** of each other. There can also be problems where it is not going to have an end like certain robots that are used for personal assistance which do not terminate until an external signal from environment puts it in termination state. These are called **continuous** tasks.

In episodic tasks we can calculate the returns which is a total sum of its rewards till the termination but in continuous tasks as there is no termination, the rewards add up to **infinity** while calculating the returns. So we introduce a discount factor gamma**(ɤ)** to calculate the returns in continuous tasks by discounting it. It has it values from 0–1. It plays a crucial role in determining if we give importance to immediate rewards or future rewards. If **ɤ=0** then the attention lies on immediate rewards and if **ɤ=1** then on future rewards.

![walking]({{ site.baseurl }}/assets/images/RL1-5.jpg)


If we have a problem statement which says you get a reward of 1 for performing certain action for next k time steps with a discount factor 0.8 and 0.2 then the returns would be

![walking]({{ site.baseurl }}/assets/images/RL1-6.jpg)

>  We see that Gt with **ɤ=0.8** is yielding a good return even in future but for **ɤ=0.2 **the returns are high only in the immediate time step and in future it is almost tending to zero. So based on the problem statement we can set **ɤ **that facilitates to decide the importance of either **immediate** or **future **rewards.

We now know an agent changes its states and gets a reward for that transition. Lets check an example to understand in detail:

![walking]({{ site.baseurl }}/assets/images/RL1-7.jpg)

Consider a situation where a student is an agent and he have four states Home, School, Class, Movie and a discount factor of 0.8. The probabilities of transitioning from one state to other is shown in blue boxes and rewards are shown in brown boxes. Agent might have many episodes i.e., a sequence of traversing through states. For example,

1. Home -> School -> Class -> Home -> Terminate — Lets calculate the returns of state Home. G = 3 + 5*0.8 + 5*0.8*0.8 = 10.2
2. Home -> School -> School -> Movie -> Home -> Terminate — Returns in this episode is G = 3 + 2*0.8 + (-10)*0.8*0.8 + 3*0.8*0.8*0.8 = — 0.264

It is clear that episode 1 results in high return than episode 2. So it would be feasible to follow it. Returns is a significant concept as it can decide the agents optimal path.

## Markov Reward Process(MRP):

MRP is a Markov process setting which specifies a reward function and a discount factor **ɤ**. It is formally represented using the tuple (S, P, R, γ) which are listed below:

* S : A finite state space.

* P : A transition probability model that specifies P(s`|s).

* R : A reward function that maps states to rewards (real numbers) R(s) = E[ri |si = s] , ∀ i = 0, 1, . . . .(E here is expected value and i is every time step)

* γ: Discount factor — lies between 0 and 1.

**State Value Function**: State Value function Vt(s) is the expected sum of returns starting from state s at a time t.

![walking]({{ site.baseurl }}/assets/images/RL1-8.jpg)

Simply put, value function denotes how good it is for an agent to be in that particular state. Transitioning between states that result in a high reward during episodes is an optimal MRP. We have different methods to evaluate V(s). They are

1. Monte-Carlo Simulation Method: In MRP, for each episode returns are calculated and are averaged. So State Value function is calculated as Vt(s)=Sum(Gt)/Number of episodes.

2. Analytic Solution: If the number of time steps are infinite, then we cannot calculate sum or average of returns. In this process, we define γ<1 and State value function is equal to sum of Immediate Reward(reward obtained for transitioning from State s to s') and discounted sum of future rewards. The equation can be represented in Matrix form as V=R + γPV. Rearranging gives V by multiplying inverse matrix of (I − γP) with R.

![walking]({{ site.baseurl }}/assets/images/RL1-9.jpg)

3. Iterative Solution: In this method we calculate Value Function at time step t by iterating through its previous value functions at time t-1,t-2 etc., We will look into this in depth soon.

## Markov Decision Process (MDP):

An MDP is simply an MRP but with the specifications of a set of actions that an agent can take from each state. It is represented a tuple (S, A, P, R, γ) which denotes:

* S : A finite state space.

* A : A finite set of actions which are available from each state s.

* P : A transition probability model that specifies P(s`|s).

* R : A reward function that maps states to rewards (real numbers) R(s) = E[ri |Si = s, Ai=a] , ∀ i = 0, 1, . . . .(state s, action taken a, E here is expected value and i is every time step)

* γ : Discount factor — lies between 0 and 1. An episode of a MDP is thus represented as (s0, a0, r0, s1, a1, r1, s2, a2, r2, . . .).

In MRP, we have transition probability of going from one state to the other. In MDP, the notation is slightly changed. We define transition probability with respect to action as well P(Si+1|Si , ai). An example of robot transitioning between different states also depends on the action it takes if its moving forward, left, right or halted. All the other notations of returns(Gt), discount factor(γ) are exactly the same as referred in MRP.

## Policy and Q-Function:

![walking]({{ site.baseurl }}/assets/images/RL1-10.jpg)

Suppose there is a robot which is currently at a state S1, it can take actions left or right with probabilities al and ar respectively for taking left or right which lands in two different states S2 and S3. Value function and rewards of that state are also mentioned and discount factor = 0.8

Let us calculate the value function of S1:
V1= al(R+γ*V2) + ar(R+γ*V3) 
 = al(2+ 0.8*10) + ar(1+ 0.8*15) 
 = al*10 + ar*13.5

If a1 = 0.2 and ar = 0.8, then **V1 = 12.8** and if a1 = 0.8 and ar = 0.2 then **V1 = 10.7** Clearly giving more probability to take the action **ar** will give a better result in terms of value of the state S1.
>  To evaluate how good it is to transition to a state is we use **value function** but to determine how good is it to take an **action** ‘a’ from this state? This is where the concept of **Policy** sets in.

### Policy π:

Policy is a **decision making** mechanism in MDP that maps states to actions. For a policy π, if at time t the agent is in state s, it will choose an action a with probability given by π(a|s).


![walking]({{ site.baseurl }}/assets/images/RL1-11.jpg)

Given a policy **π** how can we evaluate if its good or not? The intuition is same as in MRP, we calculate the expected rewards. We can define as below:

**State Value Function** (how good is it to transition to a state): Value function at a given state s of an agent, is the expected returns obtained by following a policy **π** and reaching to a next state, until we reach a terminal state.


![walking]({{ site.baseurl }}/assets/images/RL1-12.jpg)

### State-Action Value Function or Q — Function:

The state-action value function for a state s and action a is defined as the expected return starting from the state St = s at time t and taking the action At = a, and then subsequently following the policy π. It is written mathematically as

![walking]({{ site.baseurl }}/assets/images/RL1-13.jpg)

So this tells us the value of performing an action a in state s following policy π.

These are just building blocks of MDP in RL. There are lot more concepts like Bell Man Backup Operator, Finding Optimal Value functions and Policies and Dynamic Programming etc., Lets check them out in my next blog.

Hope this article has pushed your understanding of RL to some level up.

Thanks for your time !
