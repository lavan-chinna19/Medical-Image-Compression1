| group | configuration | bd_psnr_vs_legacy | mean_bpp_q50 | mean_psnr_q50 |
| --- | --- | --- | --- | --- |
| baseline | Legacy DWT (bior4.4, L4, theta=0, bias=0.25, legacy=True) | 0.0000 | 0.2301 | 36.1478 |
| a_weights | weight_scheme=default | -0.0023 | 0.2302 | 36.1500 |
| a_weights | weight_scheme=uniform | -0.0093 | 0.3218 | 38.3900 |
| a_weights | weight_scheme=fine_1.15 | 0.0726 | 0.2754 | 37.4500 |
| a_weights | weight_scheme=fine_0.9 | -0.1989 | 0.3710 | 39.1300 |
| b_deadzone | theta=0.50, bias=0.00 (weights=fine_1.15) | 0.0509 | 0.4156 | 39.7500 |
| b_deadzone | theta=0.50, bias=0.10 (weights=fine_1.15) | -0.2928 | 0.4156 | 39.3600 |
| b_deadzone | theta=0.50, bias=0.20 (weights=fine_1.15) | -0.8043 | 0.4156 | 38.8100 |
| b_deadzone | theta=0.50, bias=0.30 (weights=fine_1.15) | -1.4246 | 0.4156 | 38.1500 |
| b_deadzone | theta=0.40, bias=0.00 (weights=fine_1.15) | 0.3015 | 0.3724 | 39.4000 |
| b_deadzone | theta=0.40, bias=0.10 (weights=fine_1.15) | 0.2549 | 0.3724 | 39.3300 |
| b_deadzone | theta=0.40, bias=0.20 (weights=fine_1.15) | 0.0334 | 0.3724 | 39.0700 |
| b_deadzone | theta=0.40, bias=0.30 (weights=fine_1.15) | -0.3340 | 0.3724 | 38.6700 |
| b_deadzone | theta=0.33, bias=0.00 (weights=fine_1.15) | 0.2614 | 0.3484 | 38.9900 |
| b_deadzone | theta=0.33, bias=0.10 (weights=fine_1.15) | 0.3525 | 0.3484 | 39.0600 |
| b_deadzone | theta=0.33, bias=0.20 (weights=fine_1.15) | 0.2888 | 0.3484 | 38.9700 |
| b_deadzone | theta=0.33, bias=0.30 (weights=fine_1.15) | 0.0803 | 0.3484 | 38.7400 |
| b_deadzone | theta=0.25, bias=0.00 (weights=fine_1.15) | 0.1244 | 0.3260 | 38.4600 |
| b_deadzone | theta=0.25, bias=0.10 (weights=fine_1.15) | 0.3167 | 0.3260 | 38.6300 |
| b_deadzone | theta=0.25, bias=0.20 (weights=fine_1.15) | 0.3844 | 0.3260 | 38.6800 |
| b_deadzone | theta=0.25, bias=0.30 (weights=fine_1.15) | 0.3235 | 0.3260 | 38.6000 |
| c_context | context_mode=none | 0.3844 | 0.3260 | 38.6800 |
| c_context | context_mode=run | 0.6016 | 0.3053 | 38.6800 |
| c_context | context_mode=direct |  | 1.2336 | 38.6800 |
| d_wavelet_levels | bior2.2 L3 | 0.7701 | 0.3607 | 39.8300 |
| d_wavelet_levels | bior2.2 L4 | 0.8896 | 0.3292 | 39.2900 |
| d_wavelet_levels | bior2.2 L5 | 0.7318 | 0.3069 | 38.7300 |
| d_wavelet_levels | bior4.4 L3 | 0.2998 | 0.3376 | 39.2700 |
| d_wavelet_levels | bior4.4 L4 | 0.6016 | 0.3053 | 38.6800 |
| d_wavelet_levels | bior4.4 L5 | 0.3943 | 0.2866 | 38.0800 |
| d_wavelet_levels | db2 L3 | 0.0536 | 0.3406 | 38.8500 |
| d_wavelet_levels | db2 L4 | 0.3448 | 0.3056 | 38.2900 |
| d_wavelet_levels | db2 L5 | 0.2177 | 0.2820 | 37.7100 |
| final_benchmarks | Best Tuned DWT vs Custom DCT | 0.9181 | 0.3292 | 39.2900 |
| final_benchmarks | Best Tuned DWT vs JPEG2000 | -2.3417 | 0.3292 | 39.2900 |
