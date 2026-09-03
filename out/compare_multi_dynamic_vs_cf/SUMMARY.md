# 多动态场景 vs CacheFormer

| 场景 | 动态类型 | CF ms | pruned ms | vs CF | absL1 | PSNR | RF调用 | 速度过 | 质量过 |
|------|----------|------:|----------:|------:|------:|-----:|-------:|:------:|:------:|
| cbox | orbit_only | 269.0 | 251.8 | 1.07x | 0.0091 | 48.5 | 4/12 | PASS | PASS |
| cbox | fov_sweep | 253.1 | 128.2 | 1.97x | 0.0115 | 47.9 | 4/12 | PASS | PASS |
| cbox | roughness_orbit | 317.8 | 207.6 | 1.53x | 0.0072 | 48.5 | 4/12 | PASS | PASS |
| cbox | specular_orbit | 320.8 | 196.3 | 1.63x | 0.0081 | 48.7 | 4/12 | PASS | PASS |
| cbox | irradiance_orbit | 318.1 | 197.3 | 1.61x | 0.0072 | 48.6 | 4/12 | PASS | PASS |
| cbox | roughness_fast | 318.0 | 262.6 | 1.21x | 0.0034 | 74.1 | 8/12 | PASS | PASS |
| cbox | combined | 318.8 | 274.6 | 1.16x | 0.0085 | 71.7 | 8/12 | PASS | PASS |
| veach-mis | orbit_only | 224.1 | 91.6 | 2.45x | 0.0043 | 51.5 | 4/12 | PASS | PASS |
| veach-mis | fov_sweep | 225.9 | 91.3 | 2.47x | 0.0043 | 51.3 | 4/12 | PASS | PASS |
| veach-mis | roughness_orbit | 275.3 | 141.1 | 1.95x | 0.0045 | 50.7 | 4/12 | PASS | PASS |
| veach-mis | specular_orbit | 276.5 | 143.4 | 1.93x | 0.0041 | 51.6 | 4/12 | PASS | PASS |
| veach-mis | irradiance_orbit | 280.8 | 145.8 | 1.93x | 0.0034 | 51.4 | 4/12 | PASS | PASS |
| veach-mis | roughness_fast | 280.5 | 208.6 | 1.34x | 0.0021 | 75.1 | 8/12 | PASS | PASS |
| veach-mis | combined | 275.8 | 142.0 | 1.94x | 0.0046 | 49.6 | 4/12 | PASS | PASS |
| shader-ball | orbit_only | 429.7 | 209.0 | 2.06x | 0.0265 | 48.8 | 4/12 | PASS | PASS |
| shader-ball | fov_sweep | 429.6 | 209.4 | 2.05x | 0.0280 | 48.5 | 4/12 | PASS | PASS |
| shader-ball | roughness_orbit | 614.3 | 399.3 | 1.54x | 0.0188 | 48.5 | 4/12 | PASS | PASS |
| shader-ball | specular_orbit | 636.9 | 391.7 | 1.63x | 0.0263 | 48.8 | 4/12 | PASS | PASS |
| shader-ball | irradiance_orbit | 616.1 | 402.4 | 1.53x | 0.0198 | 48.8 | 4/12 | PASS | PASS |
| shader-ball | roughness_fast | 616.4 | 504.5 | 1.22x | 0.0071 | 74.3 | 8/12 | PASS | PASS |
| shader-ball | combined | 615.0 | 516.3 | 1.19x | 0.0225 | 72.2 | 8/12 | PASS | PASS |
| cbox-bunny | orbit_only | 272.4 | 149.1 | 1.83x | 0.0120 | 45.5 | 4/12 | PASS | PASS |
| cbox-bunny | fov_sweep | 275.8 | 142.9 | 1.93x | 0.0136 | 46.4 | 4/12 | PASS | PASS |
| cbox-bunny | roughness_orbit | 356.2 | 221.8 | 1.61x | 0.0132 | 49.5 | 4/12 | PASS | PASS |
| cbox-bunny | specular_orbit | 356.4 | 223.7 | 1.59x | 0.0119 | 42.2 | 4/12 | PASS | PASS |
| cbox-bunny | irradiance_orbit | 356.1 | 223.6 | 1.59x | 0.0096 | 45.3 | 4/12 | PASS | PASS |
| cbox-bunny | roughness_fast | 360.0 | 298.3 | 1.21x | 0.0067 | 74.2 | 8/12 | PASS | PASS |
| cbox-bunny | combined | 360.4 | 311.3 | 1.16x | 0.0113 | 73.0 | 8/12 | PASS | PASS |
| crystals | orbit_only | 152.7 | 58.5 | 2.61x | 0.0328 | 46.9 | 4/12 | PASS | PASS |
| crystals | fov_sweep | 154.0 | 58.9 | 2.62x | 0.0195 | 49.3 | 4/12 | PASS | PASS |
| crystals | roughness_orbit | 170.3 | 77.0 | 2.21x | 0.0297 | 47.3 | 4/12 | PASS | PASS |
| crystals | specular_orbit | 172.3 | 78.3 | 2.20x | 0.0207 | 47.5 | 4/12 | PASS | PASS |
| crystals | irradiance_orbit | 174.3 | 76.3 | 2.28x | 0.0247 | 46.8 | 4/12 | PASS | PASS |
| crystals | roughness_fast | 170.9 | 125.7 | 1.36x | 0.0115 | 73.4 | 8/12 | PASS | PASS |
| crystals | combined | 172.6 | 134.0 | 1.29x | 0.0124 | 73.3 | 8/12 | PASS | PASS |

- 用例数: 35
- 速度通过 (>1×CF): 35
- 质量通过 (absL1<0.08): 35
- 平均加速比: 1.740x

每用例分帧与 contact sheet 见对应子目录。