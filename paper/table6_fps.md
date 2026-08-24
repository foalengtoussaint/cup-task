# Capture rate ablation: 60 Hz vs simulated 30 Hz

Pearson against the same optical reference (60 Hz, optical windows). The markerless arm is decimated before smoothing; SmoothNet, the segmenter and every measure run at the stated rate. Trials the segmenter declines are excluded, as in Table III.

| Movement quality measure | r_s 60 | r_s 30 | d r_s | r_av 60 | r_av 30 | med. signed diff 30-60 | as % of 60 | same, 60 Hz cutoff left in | n |
|---|---|---|---|---|---|---|---|---|
| PV [mm/s] | 0.89 | 0.88 | -0.008 | 0.94 | 0.94 | -34.184 | -6.6% | -6.6% | 750 |
| Elbow angular PV [deg/s] | 0.90 | 0.91 | +0.009 | 0.98 | 0.99 | -8.616 | -8.7% | -10.1% | 750 |
| Time to PV [s] | 0.96 | 0.96 | -0.003 | 0.99 | 1.00 | +0.017 | +1.5% | +1.5% | 748 |
| Time to first PV [s] | 1.00 | 0.98 | -0.017 | 1.00 | 1.00 | +0.017 | +1.5% | +1.5% | 748 |
| Number of movement units [n] | 0.81 | 0.81 | +0.005 | 0.90 | 0.84 | +0.000 | +0.0% | +0.0% | 748 |
| Total movement time [s] | 0.99 | 0.99 | -0.008 | 1.00 | 1.00 | +0.000 | +0.0% | +0.0% | 748 |
| Interjoint coordination | 0.57 | 0.60 | +0.025 | 0.88 | 0.80 | -0.001 | -0.1% | +0.1% | 747 |
| Trunk displacement [mm] | 0.93 | 0.94 | +0.007 | 0.96 | 0.96 | +0.336 | +0.7% | +0.7% | 750 |
| Shoulder flexion [deg] | 0.97 | 0.97 | -0.001 | 0.99 | 0.99 | +0.217 | +0.5% | +0.5% | 750 |
| Shoulder flexion, drinking [deg] | 0.97 | 0.97 | +0.003 | 0.99 | 0.99 | -0.031 | -0.1% | -0.1% | 748 |
| Elbow extension [deg] | 0.98 | 0.98 | -0.001 | 0.99 | 0.99 | +0.732 | +0.5% | +0.6% | 750 |
| Shoulder abduction [deg] | 0.94 | 0.95 | +0.005 | 0.97 | 0.97 | -0.149 | -0.5% | -0.6% | 750 |
