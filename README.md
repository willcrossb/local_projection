# local_projection

## code folder:

This contains my own local projection package that is intended to have a very general structure
The user will create the shock vector and the full control vector themselves, and pass this to the locproj function

*lppy.py is the file containing the function* **locproj()** => a function to estimate local projection as in Jorda 2005

-----------------------INPUTS -------------------------------
* X: matrix containig the shock of interest +  controls
* Y: response variable of interest (must be a single vector)
* innov_idx: column index of the shock vector in X (indexes start at zero)
* horizon: the # of periods you want to know the response for
* sig_level: add as an # in (00,100): ex. 90 => function will output 90% confidence intervals

------------------------OUTPUTS -----------------------------
* irf: the dataframe of resulting response, confidence interval, and horizon

References: https://www3.nd.edu/~nmark/Climate/Jorda%20-%20Local%20Projections.pdf
## Lag-augmented_LocalProjections-master folder

This is a fork of fantastic work from Mikkel Plagborg-Møller, I am experimenting with translating that Matlab code into python. 
Both as a coding exercise, and a way of attaining a deeper understanding of the econometrics

Link to original repo: https://github.com/jm4474/Lag-augmented_LocalProjections
