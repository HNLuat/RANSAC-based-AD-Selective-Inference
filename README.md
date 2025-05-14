# Controllable RANSAC-based Anomaly Detection via Hypothesis Testing

This is the implementation of a novel method to compute valid p-value for RANSAC-based anomaly detection based on the Selective Inferences (SI) framework. The proposed method employs Divide-and-conquer and Dynamic Programming concepts to enhance the statistical power while maintaining efficient computational time.

## Installation & Requirements

This implementation has the following requirements:
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [mpmath](https://mpmath.org/)

## Reproducibility

All the figure results are saved in folder "/results". Some other results are shown in console.

- We offer a simple demonstration, which is the following jupyter notebook file:
  ```
  ex0_simple_demonstration.ipynb
  ```
  To reproduce the demonstration's results, please run the file again.

- To check the uniformity of the p-value, please run
    ```
    >> python ex1_fpr.py
    ```
- Example for computing p-value
    ```
    >> python ex2_p_value.py
    ```
## Example for computing p-value in R

To compute p-value in R, please install [reticulate](https://rstudio.github.io/reticulate/) first. Then, replacing ```"my_r_env"``` in line 3 of ```ex3_p_value_R.R``` with the environment that already install numpy and mpmath. Then, to compute p-value, please run  
    ```
    >> Rscript ex3_p_value_R.R
    ```
