# Table I — Kinematic-trajectory agreement (MMC BA+SmoothNet vs OMC)

Median [IQR] across all trials, per trajectory, split by arm. RMSE/bias in the trajectory's unit; r = Pearson; lag from wrist-speed cross-correlation.

| Kinematic | Metric | Unaffected arm | Affected arm |
|---|---|---|---|
| End-effector velocity | r | 0.75 [0.65, 0.84] | 0.77 [0.71, 0.82] |
|  | RMSE (mm/s) | 141.26 [123.01, 173.38] | 133.58 [116.44, 156.92] |
|  | Bias (mm/s) | -24.98 [-30.07, -15.12] | -23.90 [-31.77, -12.13] |
|  | Time lag (s) | 0.02 [0.00, 0.13] | 0.03 [0.00, 0.15] |
| Elbow angular velocity | r | 0.75 [0.65, 0.84] | 0.74 [0.65, 0.81] |
|  | RMSE (deg/s) | 28.28 [23.31, 35.12] | 26.68 [21.11, 32.93] |
|  | Bias (deg/s) | 2.02 [0.95, 3.03] | 2.31 [1.28, 3.19] |
|  | Time lag (s) | 0.02 [0.00, 0.13] | 0.03 [0.00, 0.17] |
| Elbow extension | r | 0.96 [0.91, 0.97] | 0.96 [0.94, 0.98] |
|  | RMSE (deg) | 9.06 [7.09, 11.84] | 9.18 [7.22, 10.95] |
|  | Bias (deg) | 0.55 [-2.12, 4.20] | -0.31 [-5.13, 3.14] |
|  | Time lag (s) | 0.02 [0.00, 0.13] | 0.03 [0.00, 0.17] |
| Shoulder flexion | r | 0.87 [0.81, 0.93] | 0.90 [0.77, 0.93] |
|  | RMSE (deg) | 7.79 [5.29, 13.37] | 8.14 [5.40, 11.73] |
|  | Bias (deg) | -2.87 [-8.82, 4.67] | -1.57 [-11.93, 3.01] |
|  | Time lag (s) | 0.02 [0.00, 0.13] | 0.03 [0.00, 0.17] |
| Shoulder abduction | r | 0.50 [0.17, 0.75] | 0.70 [0.56, 0.84] |
|  | RMSE (deg) | 4.78 [3.75, 7.39] | 5.19 [3.45, 6.14] |
|  | Bias (deg) | -8.77 [-11.96, -6.40] | -7.53 [-10.57, -5.33] |
|  | Time lag (s) | 0.02 [0.00, 0.13] | 0.03 [0.00, 0.17] |
| Trunk displacement | r | 0.86 [0.61, 0.94] | 0.93 [0.66, 0.96] |
|  | RMSE (mm) | 7.75 [5.73, 10.64] | 7.32 [6.17, 10.29] |
|  | Bias (mm) | 8.93 [5.14, 14.37] | 10.65 [7.21, 14.18] |
|  | Time lag (s) | 0.02 [0.00, 0.13] | 0.03 [0.00, 0.17] |
