# Table I — Kinematic-trajectory agreement (MMC BA+SmoothNet vs OMC)

Median [IQR] across all trials, per trajectory, split by arm. RMSE/bias in the trajectory's unit; r = Pearson; lag from the stacked multi-signal criterion. Both sides derived from keypoints by the same function.

| Kinematic | Metric | Unaffected arm | Affected arm |
|---|---|---|---|
| End-effector velocity | r | 0.98 [0.98, 0.99] | 0.98 [0.97, 0.98] |
|  | RMSE (mm/s) | 37.01 [28.76, 45.03] | 42.51 [33.68, 48.42] |
|  | Bias (mm/s) | -5.68 [-8.62, -1.69] | -8.91 [-12.59, -3.65] |
|  | Time lag (s) | 0.02 [0.00, 0.10] | 0.02 [0.00, 0.08] |
| Elbow angular velocity | r | 0.97 [0.96, 0.98] | 0.96 [0.95, 0.97] |
|  | RMSE (deg/s) | 11.70 [9.79, 13.90] | 11.84 [10.07, 13.63] |
|  | Bias (deg/s) | 4.03 [3.01, 5.06] | 3.84 [2.67, 5.59] |
|  | Time lag (s) | 0.02 [0.00, 0.10] | 0.02 [0.00, 0.08] |
| Elbow extension | r | 1.00 [0.99, 1.00] | 1.00 [0.99, 1.00] |
|  | RMSE (deg) | 4.16 [3.65, 4.73] | 4.24 [3.24, 5.15] |
|  | Bias (deg) | 7.44 [4.61, 8.98] | 8.08 [3.05, 11.07] |
|  | Time lag (s) | 0.02 [0.00, 0.10] | 0.02 [0.00, 0.08] |
| Shoulder flexion | r | 0.96 [0.92, 0.98] | 0.97 [0.93, 0.98] |
|  | RMSE (deg) | 3.26 [2.59, 4.07] | 2.96 [1.65, 3.97] |
|  | Bias (deg) | -5.04 [-8.78, -3.47] | -3.17 [-5.99, -0.91] |
|  | Time lag (s) | 0.02 [0.00, 0.10] | 0.02 [0.00, 0.08] |
| Shoulder abduction | r | 0.78 [0.62, 0.91] | 0.88 [0.77, 0.94] |
|  | RMSE (deg) | 2.33 [1.79, 4.16] | 2.19 [1.74, 3.06] |
|  | Bias (deg) | -5.98 [-7.95, -2.88] | -1.95 [-7.33, 0.09] |
|  | Time lag (s) | 0.02 [0.00, 0.10] | 0.02 [0.00, 0.08] |
| Trunk displacement | r | 0.96 [0.89, 0.98] | 0.98 [0.94, 0.99] |
|  | RMSE (mm) | 3.96 [3.14, 6.11] | 4.50 [3.73, 6.77] |
|  | Bias (mm) | 1.68 [-0.50, 3.67] | 0.71 [-2.70, 4.06] |
|  | Time lag (s) | 0.02 [0.00, 0.10] | 0.02 [0.00, 0.08] |
