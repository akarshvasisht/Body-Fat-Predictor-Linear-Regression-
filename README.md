# Body Fat Predictor – Linear Regression
## Summary
Percentage of body fat, age, weight, height, and ten body circumference measurements (e.g., abdomen) are recorded for 252 men. Body fat, one measure of health, has been accurately estimated by an underwater weighing technique. Fitting body fat to the other measurements using multiple regression provides a convenient way of estimating body fat for men using only a scale and a measuring tape. In this project, I will be looking at the [bodyfat dataset](http://jse.amstat.org/v4n1/datasets.johnson.html) and building a regression model by synthesizing the beta values.

## Program Specification
I used the bodyfat dataset [(bodyfat.csv)](https://github.com/akarshvasisht/Body-Fat-Predictor-Linear-Regression-/blob/main/bodyfat.csv) for this project. The following functions were implemented to arrive at a prediction for the body dat percentage. 

1. <ins>**get_dataset(filename)**   </ins> — takes a filename and returns the data as described below in an n-by-(m+1) array
2. <ins>**print_stats(dataset, col)**   </ins> — takes the dataset as produced by the previous function and prints several statistics about a column of the dataset; does not return anything
3. <ins>**regression(dataset, cols, betas)**   </ins> — calculates and returns the mean squared error on the dataset given fixed betas
4. <ins>**gradient_descent(dataset, cols, betas)**   </ins> — performs a single step of gradient descent on the MSE and returns the derivative values as an 1D array
5. <ins>**iterate_gradient(dataset, cols, betas, T, eta)**   </ins> — performs T iterations of gradient descent starting at the given betas and prints the results; does not return anything
6. <ins>**compute_betas(dataset, cols)**   </ins> — using the closed-form solution, calculates and returns the values of betas and the corresponding MSE as a tuple
7. <ins>**predict(dataset, cols, features)**   </ins> — using the closed-form solution betas, return the predicted body fat percentage of the give features.
8. <ins>**synthetic_datasets(betas, alphas, X, sigma)**   </ins> — generates two synthetic datasets, one using a linear model and the other using a quadratic model.
9. <ins>**plot_mse()**   </ins> — fits the synthetic datasets, and plots a figure depicting the MSEs under different situations.

