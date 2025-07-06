📊 Anomaly Detection in Transactions - v1.0.0

This project implements a statistical anomaly detection system to identify suspicious or unusual transactions using the Gaussian distribution.
It estimates the mean and variance of features in the dataset, computes probability scores for each transaction, and determines an adaptive threshold based on the largest gap between sorted probability values. Transactions with probabilities below this threshold are flagged as anomalies.
