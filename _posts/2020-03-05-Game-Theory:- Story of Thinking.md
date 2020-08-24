---
layout: post
title:  "Game Theory: Story of Thinking"
author: Kowshik
categories: [Game Theory]
image: assets/images/GT2.jpg
tags: Game Theory, tutorial
---
In this blog, we will discuss about thinking, which is inevitable process before any decision making. We will lay theoretical framework for this thinking process. All decision making problems involves player, alternatives to choose, consequences of the outcome and preferences of those consequences. These consequences can be borne by the player himself or other players. ( Note that the consequences of decisions of other players can influence your payoff as well).

**Actions**: Alternatives from which player can choose<br/>
**Outcomes**: consequences from player’s actions<br/>
**Preferences**: how the player ranks the set of possible outcomes<br/>

The above described features quantify a **decision problem**, these decision problems can be as trivial as choosing attire in morning and as weightier as drawing peaceful frontiers in the conflict land.

Let’s use a simple example to elucidate the theory we are going to explore in coming sections

Consider a case, where you are asked to choose your desert and you are given choice of milkshake and ice cream. We can define your set of actions as A = {a, b}, where a denotes the choice of milkshake and b denotes the choice of ice cream. we will denote the set of outcomes by X = {x, y}, where x denotes drinking milk shake and y denotes eating ice cream.

## Preference Relations

Now lets include preferences, we will now introduce term **Preference Relations**. For example you prefer drinking milkshake to eating ice cream. Then we will write **x >∼ y**, which should be read as **“x is at least as good as y.”** From now, This is how we express players preferences.

Lets also include 2 other important relations:
1. *Strict Preference Relation*: x > y, for **“x is strictly better than y,”**<br/> 
2. *Indifference Relation*: x ∼ y, for **“x and y are equally good.”**<br/>

Looks like we defined decision problem, its features and also discussed preference in detail. But, everything looks trivial when a simple example is taken. Imagine having continuous action space, where you are asked to choose a rational number between a given range or imagine having a probabilistic outcomes, where the outcomes to your actions are not certain but follows a distribution.

## Assumptions

Before laying down theorems, we will make two important assumptions about the player’s ability to think through the this decision problem.

1. *The Completeness Axiom*: Any two outcomes x, y ∈ X can be ranked by the preference relation, so that either **x >∼ y or y >∼ x**. This way we are enforcing the player to take preferences. Given two outcomes, player should prefer one over other.<br/>

2. *The Transitivity Axiom*: The preference relation **>∼** is transitive: for any three outcomes **x, y, z ∈ X, if x > ∼ y and y >∼ z then x >∼ z**.<br/>

If you observe closely, with these two assumptions, we are enforcing player to definitely prefer one outcome given all possible outcomes. This way we can ensure that player behaves consistently.


## Payoff Functions

In this section, we will try to quantify this preference. let's take a example where you can take walk, bus and cab to your school and you will have to pay fine for your late arrival. Walking costs you 0$ but you will end up paying 10$ fine, bus costs you 2$ but you will have to pay 4$, car costs you 15$ and you reach just in time. 

**Actions A**: {Walk, Bus, Cab}<br/>
**Outcomes X** : {-10$, -6$, -15$}<br/>

if you prefer the outcome which costs you the least, then -6$ > -10$ >-15$. Hence you should choose Bus alternative in possible actions A. In this way, we can define the profit function. Every action a ∈ A yields a profit π(a). Then we can just look at the profit from each action and choose an action that maximises profits.

In line with above example, let define

**Payoff Function u** :X→R represents the preference relation >∼ if for any pair x, y ∈ X, u(x) ≥ u(y) if and only if x >∼ y.

We can define payoff functions to make preferences closer to the realistic situations. Rational being always chooses actions that maximise his well-being as defined by his payoff function over the resulting outcomes, for this to happen, the rational being is completely aware of all the features of the decision problem he is encountering.

A player facing a decision problem with a payoff function v(.) over actions is rational if he chooses an action a ∈ A that maximises his payoff. That is, a∗ ∈ A is chosen if and only if v(a∗) ≥ v(a) for all a ∈ A.

Let’s discuss an example with continuous action space. There is a one-kg cake, so your action set is A = [0, 1], where a ∈ A is how much you cake you can eat. Your preferences are represented by the following payoff function over actions:

![walking]({{ site.baseurl }}/assets/images/GT3.jpg)

you must maximise your payoff, in order to do that take a differential and equate it to zero. we obtain<br/>
2 − 8a = 0 and a = 0.25<br/>
implies that in-order to maximise your payoff, you must eat 250 grams considering how much cake to eat as a decision problem.

## Stochastic Outcomes

Not always the outcomes of the actions taken by the player are certain, these outcomes can be random ( stochastic). In order to fit this stochasticity into the theoretical framework, we must introduce the concept of stochastic outcomes and probabilities so that player can compare uncertain consequences in a meaningful way.<br/>
There is an uncertain element attached to the action taken by the player. In order to capture the uncertainty in a precise way, we will use the well understood notion of randomness, or risk, as described by a random variable. Using random variables is a standard mathematical way of consistently describe situations where randomness is involved.

We can utilise a decision tree to describe the player’s decision problem that includes uncertainty. Take a decision problem, where player’s action space is <br/>
A: {g,s} and player gets 10 units payoff with probability 0.75 and 0 units with probability 0.25, if player choose action g . The probabilities are 0.5:0.5 if player chooses action s to get payoff 10:0. We can describe this decision problem as follows:<br/>

![walking]({{ site.baseurl }}/assets/images/GT4.jpg)

We will try to point out some obvious deductions from the above decision problem and then generalise it for a given decision problem.

1. The probability of each outcome cannot be a negative number<br/>
2. The sum of all probabilities over all outcomes must add up to 1<br/>

It is also important to note that the probability is conditional on the action taken by the player. Hence, given an action a ∈ A, the conditional probability that xⱼ ∈ X occurs is given by <br/>
p(xⱼ|a), where p(xⱼ|a) ≥ 0, and ⅀p(xk|a) = 1 for all a ∈ A.

**Definition**: A decision problem with outcomes X = {x₁, x₂, . . . , xn} is defined as a probability distribution p = (p(x₁), p(x₂), . . . , p(xn)), where p(xⱼ) ≥ 0 is the probability that xⱼ occurs and ⅀p(xᵢ ) = 1.

Note that our trivial decision problem of certain consequences to a action can be considered as a decision problem in which the probability over outcomes after any choice is equal to 1 for some outcome and 0 for all other outcomes. We call such a lottery a degenerate lottery. You can now see that decision problems with no randomness are just a very special case of those with randomness.

> Question: Try solving this decision problem: What are the probabilities of outcome g and outcome s?
we will discuss the solution in the next blog. ( Here you see more than one nodes for outcome g, do not get confused about it. The randomness unfolds over time, for a given action, the distribution of payoff can change with time. This decision tree depicts exactly the same thing.)

![walking]({{ site.baseurl }}/assets/images/GT5.jpg)


## Continuous Outcomes

we will go a step further to describe random variables over continuous outcome sets, that is outcome as discussed might not be discrete. To start, consider the following example. Farmer had a orange farm, the yield depends on watering and temperature. The supply water can vary from 0 to 100 gallons continuously and the temperature can also vary continously. This implies that your final yield, given any amount of water, will also vary continuously.

In above decision problems, we describe the uncertainty with a discrete probability, but in this continuous outcome set we will have describe uncertainty with a *cumulative distribution function (CDF)*

**Definition**: A simple probability over an interval X is given by a cumulative distribution function F :X→[0, 1], where F(x) = Pr{x ≤x*} is the probability that the outcome is less than or equal to x*.

Lets understand this little closely: it is somewhat meaningless to talk about the probability of growing a certain exact weight of oranges. However, it is meaningful to talk about the probability of being below a certain weight x*, which is given by the **CDF F(x*)**, or similarly the probability of being above a certain weight x*. **CDF F(10)** gives the probability the orange yield is less than 10kg.

Till now we have only discussed how to represent randomness , we can now move along to see how our decision-making player evaluates these random outcomes.

## Decision Making

If the outcomes are certain, then the decision making is simple. lets consider this example of decision problem.

![walking]({{ site.baseurl }}/assets/images/GT6.jpg)

The player has choices b,m,a,s. In this example the payoff of each outcome is also given at the end node. Since the player always prefers outcomes that has highest payoff. The preference order is s>a>m>b, The player chooses action s. Pretty straight forward isn’t it.

Now lets consider another decision problem which involves random outcomes. Unlike above case, it is not straight forward as it involves **stochastic outcomes.**

![walking]({{ site.baseurl }}/assets/images/GT7.jpg)

Intuitively, it seems that the two probabilities that follow g and s are easy to compare. Both have the same set of outcomes, a profit of 10 or a profit of 0. The choice g has a higher chance at getting the profit of 10, and hence we would expect rational player to choose g.


Since the payoffs are 10,0 with different probability for both the outcomes, we can simply judge which outcome is more preferred based on just probability. Now lets consider a less obvious revision of above decision problem.

![walking]({{ site.baseurl }}/assets/images/GT8.jpg)

As you can easily observe, the revision is that if player chooses outcome g, the with probability 0.25 player will borne a -1(negative payoff). Now the comparison is not as obvious as earlier. In order to tackle this decision problem we must calculate the payoff that can be expected from an out-come, lets introduce the concept of **expected payoff**


**Definition**: Let u(x) be the player’s payoff function over outcomes in X = {x₁, x₂, . . . , xn }, and let P= (p₁, p₂, . . . , pn) be a lottery over X such that Pⱼ = Pr{x = xⱼ}. Then we define the player’s expected payoff from the lottery P as <br/> 
E[u(x)|p]= ⅀pⱼ.u(xⱼ) = p₁.u(x₁) + p₂.u(x₂) + . . .


Using above definition, if we try to solve the revised decision problem

By choosing g, the expected payoff to the player is<br/>
v(g) = E[u(x)|g]= 0.75 *(9) + 0.25*(−1) = 6.5<br/>
In contrast, by choosing s his expected payoff is<br/>
v(s) = E[u(x)|s]= 0.5*(10) + 0.5*(0) = 5.<br/>
The expected payoff from s is still 5, while the expected payoff from g is 7.5, so that g is his best choice.<br/>

Let’s continue this decision evaluation case in continuous action space by using the introduced topic of cumulative distribution function in the next section.

Imagine outcomes can be any one of a continuum of values distributed on some interval X. We will start with evaluation of decision problem involving continuous outcomes.

We will try to estimate expected payoff. Using the concept of cumulative distribution functions which is again introduced in the last blog, we will define expected payoff in continuum case as follows:

Definition Let u(x) be the player’s payoff function over outcomes in the interval X with a lottery given by the cumulative distribution F(x), with density f(x). Then we define the player’s expected payoff:

Also we must keep in mind that the density function f(x) is just a derivative for CDF F(x).

![walking]({{ site.baseurl }}/assets/images/GT9.jpg)


After discussing the discrete and continuous action and outcome cases, it is bit settled that the rational player, who understands the stochastic consequences of each of his actions, will choose an action that offers him the highest expected payoff.


Let’s illustrate an another example of maximizing expected payoff with a finite set of actions and outcomes, Imagine that you have been working for a company and you are taking a decision whether or not join MBA. MBA fees and coaching costs you 10L(opportunity costs included)<br/>
if labour marker is strong and economy is bullish . your income value from having an MBA is 32L, while your income value from your current job is 12L.

if labour marker is average and economy is monotonous . your income value from having an MBA is 16L, while your income value from your current job is 8L. if labour marker is weak and economy is bearish. your income value from having an MBA is 12L, while your income value from your current job is 4L. lets assume the labor market will be strong with probability 0.25, average with probability 0.5, and weak with probability 0.25.
The decision is :Should you pursue the MBA?

Lets illustrate this decision problem in decision tree:

![walking]({{ site.baseurl }}/assets/images/GT10.jpg)

Note that, if player chooses to pursue MBA: then we subtract the cost of the degree from the income benefit in each of the three states of nature. Let’s calculate the expected payoff from each action. By evaluating the expected payoff values we can tell which outcome is more preferred.

![walking]({{ site.baseurl }}/assets/images/GT11.jpg)


By looking at the expected payoff values, the rational player would choose to pursue MBA.

Till now, a single player is involved in decision problem. In the next blogs in this series, we will discuss the multi player scenario.

**Thanks You :)**

