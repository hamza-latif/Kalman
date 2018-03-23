"""
KalmanFilter.py

This file implements a basic kalman filter class
"""

import numpy as np


class KalmanFilter:
    def __init__(self, A, B, H, x_k, P_k):
        self.A = A
        self.B = B
        self.H = H
        self.x_k = x_k
        self.P_k = P_k

    def predict(self, Q_k, u_k_minus_1):
        self.x_k_prior = self.A * self.x_k + self.B * u_k_minus_1
        self.P_k_prior = self.A * self.P_k * self.A.transpose() + Q_k

    def correct(self, R_k, z_k):
        K_k = self.P_k_prior * self.H.transpose() * np.linalg.inv(self.H * self.P_k_prior * self.H.transpose() + R_k)
        self.x_k = self.x_k_prior + K_k * (z_k - self.H * self.x_k_prior)
        self.P_k = (np.identity(self.P_k.shape[0]) - K_k * self.H) * self.P_k_prior

    @property
    def getPrediction(self):
        return self.x_k_prior
