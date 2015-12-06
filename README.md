# dantzigADMM
Implementation of the Dantzig Selector for Apache Spark

The Dantzig selector (Candes & Tao, 2007) is a method for learning linear models and high-dimensional model selection. It is the solution to

![Latex of model](https://www.dropbox.com/s/soayi0cqf97p5yd/dantzig.png?dl=1)

Where X is the model matrix, and y vector of dependent variables. This software finds the solution using an ADMM algorithm (Boyd et al., 2011). 
For more details see the vignette [here](https://scholar.google.com/scholar?cluster=4526624859438373542&hl=en&as_sdt=0,14).

This implementation is for the data-distributed setting. It is not yet suitable for when the number of predictors is extremely large.
