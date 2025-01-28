---
layout: post
title: Deep Neural Network Playground
sitemap: false
---

I'm trying to simulate some binary data that a Bayesian Deep Neural Network (BDNN?) will do a good job of predicting. This has proven difficult because 1) it's hard (for me!) to develop a complicated function of the covariates that would make sense to use a BDNN on and 2) to actually fit the BDNN such that it gives good performance.

While I was poking around on the internet, I came across [this](https://playground.tensorflow.org/) Deep Neural Network Playground from the folks at tensorflow, which I had seen previously in a class. It's nice because you don't need to install anything and it has a nice graphical interface to play around with. You can see how adding network layers helps predictions, how adding nodes changes predictions, and can even play around with all the parameters. It also gives a sense to what might charitably be called the "art" of machine learning since you have to tune all of these parameters to get things to work well.

Have fun! And hint: decrease the learning rate as you go.