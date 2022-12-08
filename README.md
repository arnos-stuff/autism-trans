# Computing probabilities of lognormal events for autism & transgender data

## Statistics

There's a fair bit of statistics in this project.  I'm not a statistician, but I know a thing or two about error propagation and I knew I had to implement a basic routine for this type of data.

Here's [my small blog post about this side project](https://write.as/arnov/error-propagation-in-the-logit-normal-family) and why I came to write these routines.

The idea is you have estimates for the Odds-Ratio (OR) and a CI for the OR.  You want to know the probability of a lognormal event, given the OR and the CI.

So you model every variable as lognormal (because they are) and infer the distribution of each probability estimate together with point estimates and CIs. 

## Data: Autism and gender identity

This project is based on the following data: [Nature Comms paper on Autism & Gender identity](https://www.nature.com/articles/s41467-020-17794-1).

I have [a separate blogpost on this data analysis](https://write.as/arnov/bayes-can-tell-you-youre-transgender).