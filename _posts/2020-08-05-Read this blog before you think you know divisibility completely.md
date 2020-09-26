---
layout: post
title:  Read this blog before you think you know divisibility completely !
author: kowshik
categories: [Number Theory]
image: https://cdn-images-1.medium.com/max/4800/1*SkxrzB4wtc-16Yc28W4-Wg.jpeg
tags: [features]
---

#### Understanding what numbers hide from you

![Photo by [Markus Spiske](https://unsplash.com/@markusspiske?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash]()

The intention to write this blog is not to dwell too much upon the theories. I don’t plan to confuse reader’s minds providing useless proofs to obvious things. My main motivation is to put these amazing concepts the way I’d like to teach my young-self.

## Introduction

let me quickly discuss divisibility and GCD(greatest common divisor) and also introduce few notations here:

![](https://cdn-images-1.medium.com/max/2000/1*MvIQBKkXxBheHQINUtrDGw.png)

 1. ***Divisibility:*** An integer b is divisible by a integer a, not zero, if there is an integer x such that b=ax, and we write **a|b.**

 2. ***GCD***: The greatest common divisor (gcd) of two or more integers, which are not all zero, is the largest positive integer that divides each of the integers.

For two integers a, b, the greatest common divisor of a and b is denoted **(a,b)**

## What GCD Hide From You ?

I will introduce the concept which I liked the most about GCD. Believe me, you might not believe that these concepts are truly generalised till you see a proper mathematical proof ( *which I will provide* 😉).

![](https://cdn-images-1.medium.com/max/2322/1*LTiNm-DBkR010PfCqcr9qA.png)

Wait! but it can just be another coincidence right?
Not really what you just saw is truly generalised and magnificent result.
> # ***Greatest common divisor of b and c is the least positive value of bx+cy where x,y range all over integers.***

## Proof

Lets carryout the proof in two steps. In the first step, we prove that the least positive number *l *that obtained from *b𝔁+ c𝒚 ∀ *𝔁,𝒚 ɛ𝐼 is divisible by b,c. In the second step we prove that *l* is GCD of b,c i.e *l* = (b,c)

Step 1. *l*|b and *l*|c

![](https://cdn-images-1.medium.com/max/2462/1*BC2HTI1f2Xs9qZfxTPKeng.png)

Step 2. *l* = (b,c)

![](https://cdn-images-1.medium.com/max/2314/1*HktMPqybtj2vsSIfyo0M_Q.png)

## Observations

![](https://cdn-images-1.medium.com/max/2374/1*YPWbhElnyjMfDPkYGnWNTQ.png)

## Some Extraordinary Outcomes of the Result !
> # 1. (a, b ) = ( a, b+a𝔁) for any integer 𝔁

![](https://cdn-images-1.medium.com/max/2374/1*enDBFMqy-MJ5p_sui2NG9A.png)
> # 2. GCD (n! + 1 , (n+1)!+1) = 1

![](https://cdn-images-1.medium.com/max/2448/1*U_K-MCphPGxgbtt3JmXpYQ.png)

## Few Examples !

![](https://cdn-images-1.medium.com/max/2450/1*cJnj2AEzDvBHKW_m9hY1iA.png)

## Conclusion

Understanding numbers has always been a perpetual pursuit for not just mathematicians but also for many eager minds. This blog is just an selfless and decent attempt to shed some light on the most original concepts in everyday topic like divisibility. The concepts which are introduced to the readers in this blog are generalised and provable, but there many concepts which are waiting to get discovered.

Suggestions are most welcome. Also do comment for clarifications and questions.

Thanks for your time :)

## References:

 1. *An Introduction to Theory of Number* by Iven Niven, Herbert and Hugh
