---
layout: post
title: WpProj hits CRAN!
description: >
  My first package is finally live on CRAN
sitemap: false
---

This package was quite a mess. It was the first one I’d ever decided to
do and filled with naïvetê and hubris, I thought it would be a good
idea to make a package to load all my functions. It definitely took a
long time—especially because a lot of the code was adapting someone
else’s C++ code to run for my purpose!—but I think it’s not too bad.
There are some places I see that would be obvious places to improve the
models but required more work than I currently have time to put in,
unfortunately.

## What does the package do?

Basically, this package was my round about way of re-inventing $$L_p$$
regression. Only slightly kidding. The basic idea is as follows.

Say you have some arbitrarily complex model, $$f$$, that takes
covariates $$x$$, and generates a set of predictions, call them
$$\hat{y}\in \mathcal{Y}$$, that follow a distribution $$\mu$$, with an
empirical counterpart $$\hat{\mu} $$. Now say this model is really hard
to interpret how the $$x$$’s affect the predictions. Let’s say we have
another class of models, $$g$$, that are easy to interpret. Say these
are linear models that typically have the form $$x\beta$$. We will
denote predictions from this model as $$\hat{\eta}$$ and let them have
some unspecified distribution $$\nu$$ and empirical counterpart
$$\hat{\nu}$$. (Note that since the $$x$$’s are considered fixed, the
distribution is actually coming from the $$\beta$$’s,
i.e. $$x\beta \sim \nu$$)

It’d be nice if we could use this set of interpretable models from $$g$$
to help us understand what’s happening in $$f$$. Ideally, these models
in $$g$$ would be close in some sense to $$f$$. We desire

1.  Fidelity: predictions from our $$g$$ models should be close to $$f$$
2.  Interpretability: we should be able to understand what’s happening
    in $$g$$. This also implies our models can’t have too many
    coefficients.

Let’s address each of these in turn. For 1, we need some way of ensuring
predictive distributions are close to one another. One such metric is
the $$p$$-Wasserstein distance, defined as

$$W_p(\hat{\mu}, \hat{\nu}) = \left( \inf_\pi \int \|\hat{y} - \hat{\nu}\|^p \pi(d\hat{y}, d\hat{\nu}) \right)^{1/p}.$$

Then we seek to minimize $$\inf_\hat{\nu} W_p(\hat{\mu}, \hat{\nu})^p.$$

Now, for 2. Since the parameter space of $$\hat{\nu}$$ could be quite
large if the dimensionality of $$x$$ is large, then we might not still
have an interpretable model—I’d argue that a 1000 covariate regression
model is **not**, in fact, interpretable!

Going back to our minimization problem, we want to add some kind of
penalty for large parameter distributions

$$\inf_\hat{\nu} W_p(\hat{\mu}, \hat{\nu})^p + P_\lambda (\hat{\nu}).$$

We should note that $$\hat{\nu} = x \hat{\beta}$$. So, we need someway
of reducing the dimensionality of the $$x$$’s. But fortunately, for
linear models there’s a decades old method of doing just that!

Rather than focusing on the $$x$$’s, we focus on reducing the dimensions
of $$\beta$$ using a penalty like the group Lasso:

$$P_\lambda (\beta) = \lambda \| \beta^{(1)} \|_2 + \lambda \|\beta^{(2)}\|_2 + ...$$

Finally, if we let the number of atoms in empricical distributions be
equal, then the problem will reduce to

$$\inf_\beta \sum_i \left\| \hat{y}_i - x \beta_i\right\|_p^p + \lambda \sum_j \left \|\beta^{(j)}\right\|_2.$$

Cool!

## Let’s see an example

Ok, say we have

### Estimating model

For exposition, we will keep it simple

### Interpretable model

### Performance Evaluation

- example: maybe simple normal model, bayesian
- then show predictions (how?)
- then show estimating simpler model
- show outputs

## Extensions

One obvious extension is to have arbitrary transformations of the
preditive function $$x\beta$$. This would be useful in the case where we
have something like predictions on a probability space and we want our
coefficients to be selected such that they do the best job of predicting
on that space rather than the linear predictor space:

$$\inf_\beta \sum_i \| \hat{y}_i - h(x \beta_i)\|_p^p + ... $$

This may also allow the models to have uses in other applications, which
hopefully I will be working on soon!
