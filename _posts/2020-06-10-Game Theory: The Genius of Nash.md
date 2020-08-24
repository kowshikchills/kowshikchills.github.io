---
layout: post
title: Game Theory- The Genius of Nash
author: kowshik
categories: [Game Theory]
image: assets/images/B2-1.jpg
tags: Game Theory
---

We discussed strict dominance solution concept in great detail in the last blog. Its application is limited and only applicable to some section of games( Games with strict dominant strategy). Strict dominant strategy often fails to exist. 
Lets consider **Battle of sexes** game.

![walking]({{ site.baseurl }}/assets/images/B2-2.jpg)

Dominant strategy equilibrium did not apply, because there is no dominant strategy. 
In the last blog, we discussed the concept of belief. Player will behave optimally( best response ) to their beliefs. Chris may behave optimally and go to football given his belief that Alex is going to the football game. But their beliefs can be wrong.

In this blog, we will discuss one of the most central and best known solution concept in the game theory. This overcomes many shortcoming faced by other solution concepts, this is developed by **John Nash**.

Let’s define Nash’s solution concept. Nash equilibrium is as a profile of strategies(defined in the last blog) for which each player is choosing a best response to the strategies of all other players.
Each strategy in a Nash equilibrium is a best response to all other strategies in that equilibrium
 Lets formally define nash equilibrium
 
**Definition**: The pure-strategy profile **s*= (s***₁**, s***₂**, . . . , s*n) ∈ S** is a Nash equilibrium if **s*ᵢ** is a best response to **s*₋ᵢ** , for all i ∈ N, that is,

**v**ᵢ**(s∗ᵢ , s∗₋ᵢ) ≥ v**ᵢ**(sᵢ, s∗₋ᵢ)** **for all sᵢ ∈ Sᵢ and all i ∈ N.**

Please note that s* is strategy profile, not strategy. strategy profile refers of set of actions taken by all the players in a strategic environment/game.
lets try to understand this definition by working out an example.

![walking]({{ site.baseurl }}/assets/images/B2-3.jpg)

Consider this matrix representation. Now lets write down all possible strategy profiles. 
**S** = {(L,U), (C,U),(R,U),(L,M), (C,M),(R,M),(L,D), (C,D),(R,D)}.
Now lets evaluate payoff functions vis-a-vis best response. 
if player 1 chooses U best response for player 2 is L: BR₂(U) = L
**BR**₂**(U) = L**, BR₂(M) = C, BR₂(D) = R
**BR**₁**(L) = U**, BR₁(C) = D, BR₁(R) = U
Now closely observe If player 2 chooses L, then player 1’s best response is {U}; at the same time, if player 1 chooses U, then player 2’s best response {L}. It clearly fits the definition above. 
So this is the **s*: {L, U}** Nash equilibrium. 
let’s apply Nash’s solution concept to prisoners dilemma.

![walking]({{ site.baseurl }}/assets/images/B2-4.jpg)

S = {(RS,BE), (BE,BE), (BE,RS), (RS, RS)} 
Nash equilibrium s* is (BE,BE)
I encourage readers to solve this and find out how (BE,BE) is Nash Equilibrium.
Here are the assumptions for a Nash equilibrium:
1. Each player is playing a best response to his beliefs.
2. The beliefs of the players about their opponents are correct.
We will not dig too deep into these assumptions as it can put us in mid of some philosophical discussion. 
Lets compare Nash solution concept with other solution concepts

![walking]({{ site.baseurl }}/assets/images/B2-5.jpg)

Here it easy to deduce that there is strictly dominant strategy for both players: thus strict dominance concept fails.
There is no strictly dominated strategy for any player, so iterative elimination method is not applicable.

Lets check if a pure-strategy Nash equilibrium does exist.
BR₁(L) = D, **BR**₁**(C) = M,** BR₁(R) = M
BR₂(U) = L, **BR**₂**(M) = C**, BR₂(D) = L
we find that **(M, C)** is the **pure-strategy Nash equilibrium**— and it is unique.

Solution concept is finest if it predicts or prescribes an unique strategy. It is necessary to understand if Nash equilibrium always yields unique strategy.
Lets consider the battle of sexes game.

![walking]({{ site.baseurl }}/assets/images/B2-6.jpg)

Let’s solve Nash equilibrium for this game. 
S = {(O, F), (O, O), (F, F), (F,O )}
BRa(O) = O, BRa(F) = F
BRc(O) = O, BRa(F) = F

We can clearly observe that we may not have a unique Nash equilibrium, but it usually lead to more refined predictions than those of strict dominant solution concept and iterative elimination. 
Nash equilibrium solution concept has been applied widely in economics, political science, legal studies, and even biology.
Let’s discuss an example where we can apply Nash’s solution concept to real life problem.

## Stag Hunt
>  Two individuals go out on a hunt. Each can individually choose to hunt a stag or hunt a hare. Each player must choose an action without knowing the choice of the other. If an individual hunts a stag, they must have the cooperation of their partner in order to succeed. An individual can get a hare by himself, but a hare is worth less than a stag. This has been taken to be a useful analogy for social cooperation, such as international agreements on climate change.The payoff matrix is as follows

![walking]({{ site.baseurl }}/assets/images/B2-7.jpg)

BR₁(S) = S, BR₁(H) = H
BR₂(H) = H, BR₂(S) = S
Game has two pure-strategy equilibria: (S, S) and (H, H). However, the payoff from (S, S) **Pareto dominates** that from (H, H).

If a player anticipates that the other individual is not cooperative, then he would choose to hunt a hare. But if he believes that other individual will cooperate then we would choose stag. When both individuals choose stag i.e when both believe other individual will cooperate, as a whole both of them would be better off.

## Scarce Resource

Let’s try to understand how self interested players might behave in scenario of scarce resources. Imaging there are **n** fertiliser manufacturing companies each choosing how much to produce around a fresh water lake. Each manufacturing companies degrades some amount of fresh water in that lake and uses, Lets say the total units of water in lake is K. Each player i chooses his own consumption of clean water for production, k**ᵢ** ≥ 0, and the amount of clean water left is therefore **K -⅀ki** .
The benefit of consuming an amount k**ᵢ** ≥ 0 gives player i a benefit equal to **ln(kᵢ)** to the fertiliser company, and no other player benefits from i’s choice.
Each player also enjoys consuming the remainder of the clean air, giving each a benefit ln(K −**⅀** kj). Hence the total payoff of player i is

![walking]({{ site.baseurl }}/assets/images/B2-8.jpg)

For player i from the choice k= (k₁, k₂, . . . , kn).
To compute Nash equilibrium, we need to find a strategy profile for which all players choose best-response to their beliefs about his opponent). 
That is we find strategy profile (k∗₁, k∗₂, . . . , k∗n) for which k∗**ᵢ**= BRi(k∗**₋ᵢ**) for all i ∈ N. For player I, we can get best response the by maximising the value function written above. To find ki, which maximises the value function of industry i, We can equate its derivative to zero.

![walking]({{ site.baseurl }}/assets/images/B2-9.jpg)

![walking]({{ site.baseurl }}/assets/images/B2-10.jpg)

Solving above equation gives player’s i best response.

Lets take only 2 industries case and solve this. ki(kj ) be the best response of player i.

![walking]({{ site.baseurl }}/assets/images/B2-11.jpg)

Lets plot this with k1 payoff in x axis and k2 payoff in y axis.

![walking]({{ site.baseurl }}/assets/images/B2-12.jpg)

If we solve the two best-response functions simultaneously, we find the unique Nash equilibrium, which has both players playing k₁= k₂ = K/3.

## **Mixed Strategies**

So far we discussed pure strategies, but we need to discuss the problem where player may choose to randomise between several of his pure strategies. There are many interesting applications to this kind of behaviour where player chooses actions stochastically( i.e. Instead of chooses a single strategy, player chooses a distribution of strategies). The probability of choosing any of pure strategy is nonnegative, and the sum of the probabilities of choosing any all pure strategies events must add up to one.
We will also closely observe applicability of Nash equilibrium to these mixed strategies. In fact, Nash equilibrium is applied to the games only if player chooses mixed strategies instead of pure strategies.

We start with the basic definition of random play when players have finite strategy sets **Sᵢ**:
Let **Sᵢ = {sᵢ₁, sᵢ₁, . . . , sᵢm}** be player i’s finite set of pure strategies. Define **ΔSᵢ **as the simplex of **Sᵢ**, which is the set of all probability distributions over **Sᵢ** . A mixed strategy for player i is an element **σᵢ ∈ Sᵢ**, so that
**σᵢ= {σ(sᵢ₁), σᵢ(sᵢ₂), . . . , σᵢ(sᵢm))** is a probability distribution over **Sᵢ **,
where **σᵢ(sᵢ)** is the probability that player i plays s**ᵢ** .

Now consider the example of the rock-paper-scissors game, in which S**ᵢ**= {R, P, S} (for rock, paper, and scissors, respectively). We can define the simplex as 
ΔSi ={(σ**ᵢ**(R), σ**ᵢ**(P ), σ**ᵢ**(S)) : σ**ᵢ**(R), σ**ᵢ**(P ), σ**ᵢ**(S)≥0, σ**ᵢ**(R)+σ**ᵢ**(P )+σ**ᵢ**(S)=1},

The player i and his opponents -i both choose mixed actions. It implies that player’s i belief about his opponents -i is not fixed but random. Thus a belief for player i is a probability distribution over the strategies of his opponents.

Definition: A **belief** for player i is given by a probability distribution **πᵢ∈S₋ᵢ** over the strategies of his opponents. We denote by **πᵢ(s₋ᵢ)** the probability player i assigns to his opponents playing **s₋ᵢ ∈ S₋ᵢ** .
For example in the rock-paper-scissors game, Belief of player i is represented as (**πᵢ**(R), **πᵢ**(P ), **πᵢ**(S)). We can think of σ***₋ᵢ** as the belief of player i about his opponents, π**ᵢ**, which captures the idea that player i is uncertain of his opponents.

behavior.

In pure strategy, the payoff is straight forward. In mixed strategy, to evaluate payoff we need to reintroduce the concept of **expected payoff.**

![walking]({{ site.baseurl }}/assets/images/B2-13.jpg)

The expected payoff of player i when he chooses pure strategy **sᵢ∈ Sᵢ** and his opponents choose mixed strategy **σ₋ᵢ∈ ΔS−ᵢ**
Please note that pure strategy is part of mixed strategy.

![walking]({{ site.baseurl }}/assets/images/B2-14.jpg)

When player i choose mixed strategy **σᵢ∈ ΔS**ᵢ and his opponents choose mixed strategy **σ₋ᵢ∈ ΔS₋ᵢ.**

![walking]({{ site.baseurl }}/assets/images/B2-15.jpg)

Let calculate payoff in mixed strategy scenario.

lets assume that player 2 plays σ₂(R) = 0.5
σ₂(P ) = 0.5
σ₂(S) = 0 
We can now calculate the expected payoff for player 1 if he chooses pure strategy.
V₁(R, σ₂) = 0.5*(0)+ 0.5*(-1) + 0 *(1)=-0.5
V₁(P, σ₂) = 0.5*(1)+ 0.5*(0) + 0 *(-1)= 0.5
V₁(S, σ₂) = 0.5*(-1)+ 0.5*(1) + 0 *(0)=0

**V₁(P, σ2)>V₁(S, σ2)>V₁(R, σ₂)**

To given player 2’s mixed strategy, we see a best response to player 1, which is action P.

Now let’s understand how Nash equilibrium solution concept applies to mixed strategies. It actually simpler than it looks, we just replace strategy profile with mixed strategy profile.
**Definition:** The mixed-strategy profile **σ* = (σ*₁ , σ*₂ , . . . , σ*n )** is a Nash equilibrium if for each player **σ*ᵢ **is a best response to **σ*₋ᵢ**. That is, for all i ∈ N, **vᵢ(σ*ᵢ , σ*₋ᵢ) ≥ vᵢ(σᵢ, σ*₋ᵢ). ∀ σᵢ∈ Sᵢ**.
Each mixed strategy in a Nash equilibrium is a best response to all other mixed strategies in that equilibrium.

Let’s close the discussion on mixed strategies here. We will discuss more about them in the next blog in my blog series. 
Hope you enjoy reading this blog.
Thanks :)
