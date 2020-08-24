---
layout: post
title:  "Game Theory: Contention and Cross-Effects"
author: kowshik
categories: [Game Theory]
image: assets/images/GT3-1.jpg
tags: GameTheory tutorial
---

We have so far discussed decision problems that a rational individual could face. But as we move more closer to the reality, we more often face decision problems where our well-being does depend not only on our actions but also the actions of other decision makers. Just as you are trying to optimize your decisions, so are they. 

In order to maximise your well being, you not only think of your actions but also guess what other players are doing, in order to maximise your reward (refer to the examples in my first blog). Your contenders are also no less rational than you, they take decisions in a similar way.

In essence, you and your peers are engaged in a **strategic environment** in which you have to think hard about what other players are doing in order to decide what is best for you — knowing that the other players are going through the same difficulties.
Now we see the theoretical framework, we laid for individual rational decision maker is falling apart as we introduce other decision makers into the decision problem. We now need a simple framework to capture these strategic situations. To start with, lets call these games. Lets introduce concept of static game: 

**Static game** is a game where each player chooses their action without the knowledge of the actions chosen by other players and after which these choices will result in a particular outcome, or probabilistic distribution over outcomes.
Remember the assumptions about rational choice in Game Theory: Story of Thinking blog: we need to improve more such assumptions so that each player in strategic environment can behave rationally. These assumptions help us analyse games within a structured framework. 

1) All the possible actions of all the players<br/>
2) All the possible outcomes<br/>
3) How each combination of actions of all players affe cts which outcome will materialise, and<br/>
4) The preferences of each and every player over outcomes<br/>


## Vanilla Games with Pure Strategy



It’s time to develop a formal framework to understand the strategic environment. Just like normal decision problem which involves single player(refer to my blog: Game Theory: Story of Thinking), we can introduce a decision problem where players ( More than one) have to choose actions from action space and the combinations of those such choices results in outcomes. Each player in the decision problem have preferences for these outcome.

Let’s start with decision problem with deterministic actions and deterministic outcomes. We rule out stochasticity in outcomes for now to simply lay the theoretical framework and introduce the notion of stochasticity in outcomes or probabilistic outcomes in coming blogs in this series.

As discussed above vanilla game or normal-form game consists of 1. A set of players 2. set of actions for each players and 3. set of payoff functions for each player

Payoff functions in normal-form game: which gives the payoff value for combination of actions chosen by each player in the normal game. This is defined for each player.<br/>
Now lets’ understand the **concept of strategy**. Strategy is just a plan of actions. To simplify, we will interchangeably use strategy and actions. 
Pure strategy is simply deterministic plan of actions, there is no concept of randomness involved. We will introduce stochastic strategy or mixed strategy in next blogs in this series. Let’s discuss simple example to make things more clearer for the readers.


## Prisoners Dilemma

A well known and simple example in game theory, we encounter this problem repeatedly in coming blogs.

>Two members of a criminal gang are arrested and imprisoned. Each prisoner is in solitary confinement with no means of communicating with the other. The prosecutors lack sufficient evidence to convict the pair on the principal charge, but they have enough to convict both on a lesser charge. Simultaneously, the prosecutors offer each prisoner a bargain. Each prisoner is given the opportunity either to betray the other by testifying that the other committed the crime, or to cooperate with the other by remaining silent. The possible outcomes are:<br/>
1.If A and B each “betray”(BE) the other, each of them serves two years in prison<br/>
2.If A betrays B but B “remains silent”(RS), A will be set free and B will serve three years in prison (and vice versa)<br/>
3.If A and B both remain silent, both of them will serve only one year in prison (on the lesser charge).<br/>



**Players**: N= {A,B}<br/>
**Strategic sets**: S= {BE, RS}<br/>
**Payoffs**: vA(sA,sB) be the payoff of player A and vB(sA,sB) be the payoff of player B<br/>


**vA**(BE,BE) = **vB**(BE,BE) = -2<br/>
**vA**(RS,RS) = **vB**(RS,RS) = -1<br/>
**vA**(BE,RS) = **vB**(RS,BE) = 0<br/>
**vA**(RS,BE) = **vB**(BE,RS) = -3<br/>

Its more convenient to represent these numbers in matrix representations.

![walking]({{ site.baseurl }}/assets/images/GT3-2.jpg)

**Rows**: Represent players A strategies<br/>
**Columns**: Represents player B strategies<br/>
**Matrix Entries**: payoff of A/B<br/>

To get use to this matrix representation, lets look at an another famous example in game theory


## rock-paper-scissors.

Recall that rock (R) beats scissors (S), scissors beats paper (P), and paper beats rock.<br/>
Let the winner’s payoff be 1 and the loser’s be −1, and in case of tie (choose the same action) be 0. This is a game with two<br/>

**Players**: N = {1, 2}
**Strategic sets**: S= {R, P, S}
**payoff matrix**:

![walking]({{ site.baseurl }}/assets/images/GT3-3.jpg)


I urge readers to closely analyse the payoff of each player for different combinations of action taken by each player and check if the numbers match the payoffs already described verbally above.

**Strategy Profile**: It is basically set of actions taken by player, there are 9 possible strategy profiles. For example {R,R} is one strategy profile, which implies Player 1 and 2 both decides to choose rock.

**vᵢ(s)** is payoff of a player i from a **profile of strategies** s = (s₁, s₂, . . . , sᵢ₋₁ , si, sᵢ₊₁, . . . , sn). We define strategy profile s₋ᵢ ∈ S₋ᵢ as a particular possible profile of strategies for all players who are not i.


## Solution Concept

Let’s introduce the idea of solution concept in this section. So far we stressed on representation of payoff for different combinations of unique player’s decisions in the strategic environment. These representations are useless until we apply some model to predict the decision of a given player considering the anticipated decisions taken by other rational players. We describe this model as solution concept.<br/> 
For example, solution concept can be “ players always choose the action that they think the opponent can choose” or “ player act in accordance with pareto optimal outcomes”.<br/>
Pareto optimality is a situation that cannot be modified so as to make any one individual better off without making at least one individual worse off.<br/>

This solution concept is finest if it is applied to wide variety of games, not just to a small and select family of games. This solution concept usage should ideally result in unique action. We doesn't want solution concept to result in “ take any action”.

In the next section, we will define some very important solution concepts in game theory. Let’s use our prisoners dilemma example to illustrate concepts before formally defining them.



![walking]({{ site.baseurl }}/assets/images/GT3-4.jpg)


if player choose to remain silent, the possible outcomes are -1 and -3 depends on whether player’s opponent choose to remain silent and betray respectively. if player choose to betray then the possible outcomes are 0 and -2 depends on whether player’s opponent choose to remain silent and betray respectively. Here we can easily deduce that opting to remain silent **{RS}** is worse than betraying **{BE}** for each player regardless of what the player’s opponent does. We say that such a strategy of betraying **{BE}** is dominated.

Definition: Let **sᵢ∈ Sᵢ** and **s”ᵢ∈ Sᵢ** be possible strategies for player i. We say that **s”ᵢ** is strictly dominated by **sᵢ**, if for any possible combination of the other players’ strategies, **s₋ᵢ∈ S₋ᵢ**, player i’s payoff from **s”ᵢ** is strictly less than that from **sᵢ** . That is,
**vᵢ(sᵢ, s₋ᵢ) > vi(s”ᵢ, s₋ᵢ)** for all **s₋ᵢ∈ S₋ᵢ**.
We will write **sᵢ >ᵢ s”ᵢ** to denote that **s”ᵢ** is strictly dominated by **s₋ᵢ**

We can propose a new solution concept using the definition above **Strict dominance concept** : “strictly dominant strategy is a strategy that is always the best thing you can do, regardless of what your opponents choose”

It is not difficult to use this Strict dominance concept. It basically requires that we identify a strict dominant strategy for each player and then use this profile of strategies to predict or prescribe behaviour.<br/>
In prisoners dilemma problem, each player have a strictly dominated strategy of **{BE}** Betraying, so the you would predict the players to choose betraying.

But, this solution concept only applies to a section of problems. We can easily endorse this statement by applying the Strict dominance concept to advertising game.<br/>
Two competing brands can choose one of three marketing campaigns — low (L), medium (M), and high (H) — with payoffs given by the following matrix:<br/>

![walking]({{ site.baseurl }}/assets/images/GT3-5.jpg)

It is easy to observe that there is no strictly dominant strategy for both players.( if player 2 playsM then player 1 should also play M, while if player 2 plays H then player 1 should also play H).<br/>
In absence of strictly dominant strategy, we need to conclude that the strict-dominance solution concept might not apply for all kinds of games.<br/> 
Note: To those games, where strict dominance solution concept applies. The solution it predicts or prescribes is unique i.e there can be only one strictly dominant strategy, if exists. In fact, this important intended feature is what lured us to explore this solution concept in good detail.



## Common Knowledge of Rationality


This is an important assumption which states that the structure of the game and the rationality of the players are common knowledge among the players. For example if we consider to use strict-dominance solution concept, all the players are aware each player will never play a strictly dominated strategy, they can ignore those strictly dominated strategies that their opponents will never play, and their opponents can do the same thing.

Rational player will never play a dominated strategy. We can eliminate those strategies which player will not choose for sure. 
We can iteratively eliminate the original game to a restricted game. In fact we may indeed find additional strategies that are dominated in the restricted game that were not dominated in the original game. let’s illustrate these concepts in an example. Consider the following two-player finite game:


![walking]({{ site.baseurl }}/assets/images/GT3-6.jpg)


If player 1 chooses U, the strategy with highest payoff for player 2 is L. if player 1 chooses M, the strategy with highest payoff for player 2 is R. By analysing this way, it can be deduced that there is no strictly dominant strategy for both players. If you closely observe, there exists one strictly dominated strategy for player 2. Strategy C is strictly dominated by R. This results in reduced game.

![walking]({{ site.baseurl }}/assets/images/GT3-7.jpg)


In this new matrix representation , both M and D are strictly dominated by U for player 1. This led to next level of elimination, where M,D actions are eliminated.

![walking]({{ site.baseurl }}/assets/images/GT3-8.jpg)

It is quite straight forward from here, Player 1 chooses U and player 2 chooses L as **v(L) = 3 > v(R)=2**.<br/> 
Lets try to dig little deeper into the example we discussed:<br/>

1. If a strategy sᵢ is not strictly dominated for player i then it must be that there are combinations of strategies of player i’s opponents for which the strategy sᵢ is player i’s best choice.<br/> 
This is a central concept in the game theory. The player has to choose a best strategy as a response to the strategies of his opponents. The player chooses an action considering the belief about his opponent as his/her best response.<br/>

**Definition**: The strategy sᵢ ∈ Sᵢ is player i’s best response to his opponents’ strategies s₋ᵢ ∈ S₋ᵢ if<br/> 
**vᵢ(sᵢ, s₋ᵢ) ≥ vᵢ(s”ᵢ, s₋ᵢ). ∀s”ᵢ∈ Sᵢ**

2. If sᵢ is a strictly dominated strategy for player i, then it cannot be a best response to any **s₋ᵢ ∈ S₋ᵢ**. I urge readers try to prove this proposition.


## Belief and Best Response

Suppose that sᵢ is a best response for player i to his opponents playing s’₋ᵢ. Player i will play sᵢ only when he believes that his opponents will play s’₋ᵢ. The concept of belief in central to the analysis of strategic behaviour. If a strict dominant strategy exist for player i, then regardless of his belief system player i always choses the strict dominant strategy. The player’s strictly dominant strategy is his best response independent of his opponents’ play.

![walking]({{ site.baseurl }}/assets/images/GT3-9.jpg)

If player 1 believes that player 2 is chooses strategy R then both U and D are best responses. 
So a player may have more than one best response given his belief on opponent’s choice.<br/>
Now, we have learned a bunch on decision making in strategic environments. In next blogs in this series we will delve deep into the idea mixed strategies and more solution concepts.
Thanks :)






