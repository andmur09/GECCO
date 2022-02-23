\ Model iteration_27
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  c(11,2)_rl + c(11,2)_ru + c(17,18)_rl + c(17,18)_ru + c(16,2)_rl
   + c(16,2)_ru + c(1,0)_rl + c(1,0)_ru + c(9,10)_rl + c(9,10)_ru
   + c(14,0)_rl + c(14,0)_ru + c(8,7)_rl + c(8,7)_ru + c(5,0)_rl
   + c(5,0)_ru + c(6,0)_rl + c(6,0)_ru + c(19,13)_rl + c(19,13)_ru
   + c(4,15)_rl + c(4,15)_ru + c(3,2)_rl + c(3,2)_ru + c(20,12)_rl
   + c(20,12)_ru
Subject To
 cont[0]: 2 - 16 - c(16,2)_ru <= 249.3
 cont[1]: - 2 + 16 - c(16,2)_rl <= 0
 z0: 1.434720229555237 2 - 1.434720229555237 7
   - 1.434720229555237 c(11,2)_ru + 6 lambda[0] + 6 lambda[2] + 6 lambda[3]
   + 6 lambda[4] + 6 lambda[5] + 6 lambda[6] + 6 lambda[7] + 6 lambda[8]
   + 6 lambda[9] + 6 lambda[10] + 6 lambda[11] + 6 lambda[12]
   + 6 lambda[13] + 6 lambda[14] + 6 lambda[15] + 6 lambda[16]
   + 6 lambda[17] + 6 lambda[18] + 6 lambda[19] + 6 lambda[20]
   + 6 lambda[21] + 6 lambda[22] + 6 lambda[23] + 6 lambda[24]
   + 6 lambda[25] + 6 lambda[26] <= 21.30559540889526
 z1: - 1.434720229555237 2 + 1.434720229555237 7
   - 1.434720229555237 c(11,2)_rl + 6 lambda[0] + 6 lambda[1] + 6 lambda[3]
   + 6 lambda[4] + 6 lambda[5] + 6 lambda[6] + 6 lambda[7] + 6 lambda[8]
   + 6 lambda[9] + 6 lambda[10] + 6 lambda[11] + 6 lambda[12]
   + 6 lambda[13] + 6 lambda[14] + 6 lambda[15] + 6 lambda[16]
   + 6 lambda[17] + 6 lambda[18] + 6 lambda[19] + 6 lambda[20]
   + 6 lambda[21] + 6 lambda[22] + 6 lambda[23] + 6 lambda[24]
   + 2.044727284768212 lambda[25] + 2.97662409533568 lambda[26] <= -10
 z2: - 0.9474182851729037 0 + 0.9474182851729037 18
   - 0.9474182851729037 c(17,18)_ru + 6 lambda[0] + 6 lambda[1]
   + 6 lambda[2] + 6 lambda[4] + 6 lambda[5] + 6 lambda[6] + 6 lambda[7]
   + 6 lambda[8] + 6 lambda[9] + 6 lambda[10] + 6 lambda[11] + 6 lambda[12]
   + 6 lambda[13] + 6 lambda[14] + 6 lambda[15] + 6 lambda[16]
   + 6 lambda[17] + 6 lambda[18] + 6 lambda[19] + 6 lambda[20]
   + 6 lambda[21] + 6 lambda[22] + 6 lambda[23] + 6 lambda[24]
   + 0.0103438293728071 lambda[25] + 2.522420167946224 lambda[26]
   <= 64.45328280435811
 z3: 0.9474182851729037 0 - 0.9474182851729037 18
   - 0.9474182851729037 c(17,18)_rl + 6 lambda[0] + 6 lambda[1]
   + 6 lambda[2] + 6 lambda[3] + 6 lambda[5] + 6 lambda[6] + 6 lambda[7]
   + 6 lambda[8] + 6 lambda[9] + 6 lambda[10] + 6 lambda[11] + 6 lambda[12]
   + 6 lambda[13] + 6 lambda[14] + 6 lambda[15] + 6 lambda[16]
   + 6 lambda[17] + 6 lambda[18] + 6 lambda[19] + 6 lambda[20]
   + 6 lambda[21] + 6 lambda[22] + 6 lambda[23] + 6 lambda[24]
   + 6 lambda[25] + 6 lambda[26] <= -11.36428233064898
 z4: 0.8605851979345958 0 - 0.8605851979345958 10
   - 0.8605851979345958 c(1,0)_ru + 6 lambda[0] + 6 lambda[1] + 6 lambda[2]
   + 6 lambda[3] + 6 lambda[4] + 6 lambda[6] + 6 lambda[7] + 6 lambda[8]
   + 6 lambda[9] + 6 lambda[10] + 6 lambda[11] + 6 lambda[12]
   + 6 lambda[13] + 6 lambda[14] + 6 lambda[15] + 6 lambda[16]
   + 6 lambda[17] + 6 lambda[18] + 6 lambda[19] + 6 lambda[20]
   + 6 lambda[21] + 6 lambda[22] + 6 lambda[23] + 6 lambda[24]
   + 6 lambda[25] + 6 lambda[26] <= 16.35111876075732
 z5: - 0.8605851979345958 0 + 0.8605851979345958 10
   - 0.8605851979345958 c(1,0)_rl + 6 lambda[0] + 6 lambda[1] + 6 lambda[2]
   + 6 lambda[3] + 6 lambda[4] + 6 lambda[5] + 6 lambda[7] + 6 lambda[8]
   + 6 lambda[9] + 6 lambda[10] + 6 lambda[11] + 6 lambda[12]
   + 6 lambda[13] + 6 lambda[14] + 6 lambda[15] + 6 lambda[16]
   + 6 lambda[17] + 6 lambda[18] + 6 lambda[19] + 6 lambda[20]
   + 6 lambda[21] + 6 lambda[22] + 6 lambda[23] + 6 lambda[24]
   + 1e-05 lambda[25] + 2.452774638535388 lambda[26] <= -10
 z6: 1.546790409899458 10 - 1.546790409899458 16
   - 1.546790409899458 c(9,10)_ru + 6 lambda[0] + 6 lambda[1] + 6 lambda[2]
   + 6 lambda[3] + 6 lambda[4] + 6 lambda[5] + 6 lambda[6] + 6 lambda[8]
   + 6 lambda[9] + 6 lambda[10] + 6 lambda[11] + 6 lambda[12]
   + 6 lambda[13] + 6 lambda[14] + 6 lambda[15] + 6 lambda[16]
   + 6 lambda[17] + 6 lambda[18] + 6 lambda[19] + 6 lambda[20]
   + 6 lambda[21] + 6 lambda[22] + 6 lambda[23] + 6 lambda[24]
   + 6 lambda[25] + 6 lambda[26] <= 80.85986078886312
 z7: - 1.546790409899458 10 + 1.546790409899458 16
   - 1.546790409899458 c(9,10)_rl + 6 lambda[0] + 6 lambda[1] + 6 lambda[2]
   + 6 lambda[3] + 6 lambda[4] + 6 lambda[5] + 6 lambda[6] + 6 lambda[7]
   + 6 lambda[9] + 6 lambda[10] + 6 lambda[11] + 6 lambda[12]
   + 6 lambda[13] + 6 lambda[14] + 6 lambda[15] + 6 lambda[16]
   + 6 lambda[17] + 6 lambda[18] + 6 lambda[19] + 6 lambda[20]
   + 6 lambda[21] + 6 lambda[22] + 6 lambda[23] + 6 lambda[24]
   + 2.331300128554735 lambda[25] + 3.04060659110509 lambda[26]
   <= -51.06728538283063
 z8: 0.7791195948578106 0 - 0.7791195948578106 12
   - 0.7791195948578106 c(14,0)_ru + 6 lambda[0] + 6 lambda[1]
   + 6 lambda[2] + 6 lambda[3] + 6 lambda[4] + 6 lambda[5] + 6 lambda[6]
   + 6 lambda[7] + 6 lambda[8] + 6 lambda[10] + 6 lambda[11] + 6 lambda[12]
   + 6 lambda[13] + 6 lambda[14] + 6 lambda[15] + 6 lambda[16]
   + 6 lambda[17] + 6 lambda[18] + 6 lambda[19] + 6 lambda[20]
   + 6 lambda[21] + 6 lambda[22] + 6 lambda[23] + 6 lambda[24]
   + 6 lambda[25] + 6 lambda[26] <= 12.6334242306194
 z9: - 0.7791195948578106 0 + 0.7791195948578106 12
   - 0.7791195948578106 c(14,0)_rl + 6 lambda[0] + 6 lambda[1]
   + 6 lambda[2] + 6 lambda[3] + 6 lambda[4] + 6 lambda[5] + 6 lambda[6]
   + 6 lambda[7] + 6 lambda[8] + 6 lambda[9] + 6 lambda[11] + 6 lambda[12]
   + 6 lambda[13] + 6 lambda[14] + 6 lambda[15] + 6 lambda[16]
   + 6 lambda[17] + 6 lambda[18] + 6 lambda[19] + 6 lambda[20]
   + 6 lambda[21] + 6 lambda[22] + 6 lambda[23] + 6 lambda[24]
   + 1e-05 lambda[25] + 2.37461548766725 lambda[26] <= -10
 z10: - 0.9057971014492755 0 + 0.9057971014492755 7
   - 0.9057971014492755 c(8,7)_ru + 6 lambda[0] + 6 lambda[1] + 6 lambda[2]
   + 6 lambda[3] + 6 lambda[4] + 6 lambda[5] + 6 lambda[6] + 6 lambda[7]
   + 6 lambda[8] + 6 lambda[9] + 6 lambda[10] + 6 lambda[12] + 6 lambda[13]
   + 6 lambda[14] + 6 lambda[15] + 6 lambda[16] + 6 lambda[17]
   + 6 lambda[18] + 6 lambda[19] + 6 lambda[20] + 6 lambda[21]
   + 6 lambda[22] + 6 lambda[23] + 6 lambda[24] + 6 lambda[25]
   + 6 lambda[26] <= 72.80483695652173
 z11: 0.9057971014492755 0 - 0.9057971014492755 7
   - 0.9057971014492755 c(8,7)_rl + 6 lambda[0] + 6 lambda[1] + 6 lambda[2]
   + 6 lambda[3] + 6 lambda[4] + 6 lambda[5] + 6 lambda[6] + 6 lambda[7]
   + 6 lambda[8] + 6 lambda[9] + 6 lambda[10] + 6 lambda[11] + 6 lambda[13]
   + 6 lambda[14] + 6 lambda[15] + 6 lambda[16] + 6 lambda[17]
   + 6 lambda[18] + 6 lambda[19] + 6 lambda[20] + 6 lambda[21]
   + 6 lambda[22] + 6 lambda[23] + 6 lambda[24] + 1e-05 lambda[25]
   + 2.490085179690548 lambda[26] <= -71.22282608695653
 z12: 0.7530120481927709 0 - 0.7530120481927709 13
   - 0.7530120481927709 c(5,0)_ru + 6 lambda[0] + 6 lambda[1] + 6 lambda[2]
   + 6 lambda[3] + 6 lambda[4] + 6 lambda[5] + 6 lambda[6] + 6 lambda[7]
   + 6 lambda[8] + 6 lambda[9] + 6 lambda[10] + 6 lambda[11] + 6 lambda[12]
   + 6 lambda[14] + 6 lambda[15] + 6 lambda[16] + 6 lambda[17]
   + 6 lambda[18] + 6 lambda[19] + 6 lambda[20] + 6 lambda[21]
   + 6 lambda[22] + 6 lambda[23] + 6 lambda[24] + 6 lambda[25]
   + 6 lambda[26] <= 13.35090361445783
 z13: - 0.7530120481927709 0 + 0.7530120481927709 13
   - 0.7530120481927709 c(5,0)_rl + 6 lambda[0] + 6 lambda[1] + 6 lambda[2]
   + 6 lambda[3] + 6 lambda[4] + 6 lambda[5] + 6 lambda[6] + 6 lambda[7]
   + 6 lambda[8] + 6 lambda[9] + 6 lambda[10] + 6 lambda[11] + 6 lambda[12]
   + 6 lambda[13] + 6 lambda[15] + 6 lambda[16] + 6 lambda[17]
   + 6 lambda[18] + 6 lambda[19] + 6 lambda[20] + 6 lambda[21]
   + 6 lambda[22] + 6 lambda[23] + 6 lambda[24] + 1e-05 lambda[25]
   + 2.34598929660855 lambda[26] <= -10
 z14: 1.020929045431342 0 - 1.020929045431342 15
   - 1.020929045431342 c(6,0)_ru + 6 lambda[0] + 6 lambda[1] + 6 lambda[2]
   + 6 lambda[3] + 6 lambda[4] + 6 lambda[5] + 6 lambda[6] + 6 lambda[7]
   + 6 lambda[8] + 6 lambda[9] + 6 lambda[10] + 6 lambda[11] + 6 lambda[12]
   + 6 lambda[13] + 6 lambda[14] + 6 lambda[16] + 6 lambda[17]
   + 6 lambda[18] + 6 lambda[19] + 6 lambda[20] + 6 lambda[21]
   + 6 lambda[22] + 6 lambda[23] + 6 lambda[24] + 6 lambda[25]
   + 6 lambda[26] <= 16.59520163348647
 z15: - 1.020929045431342 0 + 1.020929045431342 15
   - 1.020929045431342 c(6,0)_rl + 6 lambda[0] + 6 lambda[1] + 6 lambda[2]
   + 6 lambda[3] + 6 lambda[4] + 6 lambda[5] + 6 lambda[6] + 6 lambda[7]
   + 6 lambda[8] + 6 lambda[9] + 6 lambda[10] + 6 lambda[11] + 6 lambda[12]
   + 6 lambda[13] + 6 lambda[14] + 6 lambda[15] + 6 lambda[17]
   + 6 lambda[18] + 6 lambda[19] + 6 lambda[20] + 6 lambda[21]
   + 6 lambda[22] + 6 lambda[23] + 6 lambda[24]
   + 0.4416217724970792 lambda[25] + 2.61870687261684 lambda[26] <= -10
 z16: 1.248439450686642 13 - 1.248439450686642 16
   - 1.248439450686642 c(19,13)_ru + 6 lambda[0] + 6 lambda[1]
   + 6 lambda[2] + 6 lambda[3] + 6 lambda[4] + 6 lambda[5] + 6 lambda[6]
   + 6 lambda[7] + 6 lambda[8] + 6 lambda[9] + 6 lambda[10] + 6 lambda[11]
   + 6 lambda[12] + 6 lambda[13] + 6 lambda[14] + 6 lambda[15]
   + 6 lambda[16] + 6 lambda[18] + 6 lambda[19] + 6 lambda[20]
   + 6 lambda[21] + 6 lambda[22] + 6 lambda[23] + 6 lambda[24]
   + 6 lambda[25] + 6 lambda[26] <= 121.6103370786517
 z17: - 1.248439450686642 13 + 1.248439450686642 16
   - 1.248439450686642 c(19,13)_rl + 6 lambda[0] + 6 lambda[1]
   + 6 lambda[2] + 6 lambda[3] + 6 lambda[4] + 6 lambda[5] + 6 lambda[6]
   + 6 lambda[7] + 6 lambda[8] + 6 lambda[9] + 6 lambda[10] + 6 lambda[11]
   + 6 lambda[12] + 6 lambda[13] + 6 lambda[14] + 6 lambda[15]
   + 6 lambda[16] + 6 lambda[17] + 6 lambda[19] + 6 lambda[20]
   + 6 lambda[21] + 6 lambda[22] + 6 lambda[23] + 6 lambda[24]
   + 1.454557467861316 lambda[25] + 2.844858735895397 lambda[26]
   <= -110.7865168539326
 z18: 3.16957210776545 15 - 3.16957210776545 16
   - 3.16957210776545 c(4,15)_ru + 6 lambda[0] + 6 lambda[1] + 6 lambda[2]
   + 6 lambda[3] + 6 lambda[4] + 6 lambda[5] + 6 lambda[6] + 6 lambda[7]
   + 6 lambda[8] + 6 lambda[9] + 6 lambda[10] + 6 lambda[11] + 6 lambda[12]
   + 6 lambda[13] + 6 lambda[14] + 6 lambda[15] + 6 lambda[16]
   + 6 lambda[17] + 6 lambda[18] + 6 lambda[20] + 6 lambda[21]
   + 6 lambda[22] + 6 lambda[23] + 6 lambda[24] + 6 lambda[25]
   + 6 lambda[26] <= 239.7593026941362
 z19: - 3.16957210776545 15 + 3.16957210776545 16
   - 3.16957210776545 c(4,15)_rl + 6 lambda[0] + 6 lambda[1] + 6 lambda[2]
   + 6 lambda[3] + 6 lambda[4] + 6 lambda[5] + 6 lambda[6] + 6 lambda[7]
   + 6 lambda[8] + 6 lambda[9] + 6 lambda[10] + 6 lambda[11] + 6 lambda[12]
   + 6 lambda[13] + 6 lambda[14] + 6 lambda[15] + 6 lambda[16]
   + 6 lambda[17] + 6 lambda[18] + 6 lambda[19] + 6 lambda[21]
   + 6 lambda[22] + 6 lambda[23] + 6 lambda[24]
   + 4.209629065056486 lambda[25] + 3.459979200568478 lambda[26]
   <= -186.5768621236132
 z20: 1.33155792276964 2 - 1.33155792276964 18 - 1.33155792276964 c(3,2)_ru
   + 6 lambda[0] + 6 lambda[1] + 6 lambda[2] + 6 lambda[3] + 6 lambda[4]
   + 6 lambda[5] + 6 lambda[6] + 6 lambda[7] + 6 lambda[8] + 6 lambda[9]
   + 6 lambda[10] + 6 lambda[11] + 6 lambda[12] + 6 lambda[13]
   + 6 lambda[14] + 6 lambda[15] + 6 lambda[16] + 6 lambda[17]
   + 6 lambda[18] + 6 lambda[19] + 6 lambda[20] + 6 lambda[22]
   + 6 lambda[23] + 6 lambda[24] + 1.738292956758861 lambda[25]
   + 2.908207353643253 lambda[26] <= 12.25033288948069
 z21: - 1.33155792276964 2 + 1.33155792276964 18
   - 1.33155792276964 c(3,2)_rl + 6 lambda[0] + 6 lambda[1] + 6 lambda[2]
   + 6 lambda[3] + 6 lambda[4] + 6 lambda[5] + 6 lambda[6] + 6 lambda[7]
   + 6 lambda[8] + 6 lambda[9] + 6 lambda[10] + 6 lambda[11] + 6 lambda[12]
   + 6 lambda[13] + 6 lambda[14] + 6 lambda[15] + 6 lambda[16]
   + 6 lambda[17] + 6 lambda[18] + 6 lambda[19] + 6 lambda[20]
   + 6 lambda[21] + 6 lambda[23] + 6 lambda[24] + 6 lambda[25]
   + 6 lambda[26] <= -10
 z22: 1.302083333333333 12 - 1.302083333333333 16
   - 1.302083333333333 c(20,12)_ru + 6 lambda[0] + 6 lambda[1]
   + 6 lambda[2] + 6 lambda[3] + 6 lambda[4] + 6 lambda[5] + 6 lambda[6]
   + 6 lambda[7] + 6 lambda[8] + 6 lambda[9] + 6 lambda[10] + 6 lambda[11]
   + 6 lambda[12] + 6 lambda[13] + 6 lambda[14] + 6 lambda[15]
   + 6 lambda[16] + 6 lambda[17] + 6 lambda[18] + 6 lambda[19]
   + 6 lambda[20] + 6 lambda[21] + 6 lambda[22] + 6 lambda[24]
   + 6 lambda[25] + 6 lambda[26] <= 120.9446875
 z23: - 1.302083333333333 12 + 1.302083333333333 16
   - 1.302083333333333 c(20,12)_rl + 6 lambda[0] + 6 lambda[1]
   + 6 lambda[2] + 6 lambda[3] + 6 lambda[4] + 6 lambda[5] + 6 lambda[6]
   + 6 lambda[7] + 6 lambda[8] + 6 lambda[9] + 6 lambda[10] + 6 lambda[11]
   + 6 lambda[12] + 6 lambda[13] + 6 lambda[14] + 6 lambda[15]
   + 6 lambda[16] + 6 lambda[17] + 6 lambda[18] + 6 lambda[19]
   + 6 lambda[20] + 6 lambda[21] + 6 lambda[22] + 6 lambda[23]
   + 1.641822890533698 lambda[25] + 2.886668797509121 lambda[26]
   <= -70.234375
 cc: 2.36781e-08 lambda[0] + 0.6931472042380503 lambda[1]
   + 0.6931472042380503 lambda[2] + 0.6931472042380503 lambda[3]
   + 0.6931472042380503 lambda[4] + 0.6931472042380503 lambda[5]
   + 0.6931472042380503 lambda[6] + 0.6931472042380503 lambda[7]
   + 0.6931472042380503 lambda[8] + 0.6931472042380503 lambda[9]
   + 0.6931472042380503 lambda[10] + 0.6931472042380503 lambda[11]
   + 0.6931472042380503 lambda[12] + 0.6931472042380503 lambda[13]
   + 0.6931472042380503 lambda[14] + 0.6931472042380503 lambda[15]
   + 0.6931472042380503 lambda[16] + 0.6931472042380503 lambda[17]
   + 0.6931472042380503 lambda[18] + 0.6931472042380503 lambda[19]
   + 0.6931472042380503 lambda[20] + 0.6931472042380503 lambda[21]
   + 0.6931472042380503 lambda[22] + 0.6931472042380503 lambda[23]
   + 0.6931472042380503 lambda[24] + 4.05688274522056 lambda[25]
   + 0.0510443326193033 lambda[26] <= 0.6931471805599453
 x0: 0 = 0
 sum_lam: lambda[0] + lambda[1] + lambda[2] + lambda[3] + lambda[4]
   + lambda[5] + lambda[6] + lambda[7] + lambda[8] + lambda[9] + lambda[10]
   + lambda[11] + lambda[12] + lambda[13] + lambda[14] + lambda[15]
   + lambda[16] + lambda[17] + lambda[18] + lambda[19] + lambda[20]
   + lambda[21] + lambda[22] + lambda[23] + lambda[24] + lambda[25]
   + lambda[26] = 1
 R29: 2.36781e-08 lambda[0] + 0.6931472042380503 lambda[1]
   + 0.6931472042380503 lambda[2] + 0.6931472042380503 lambda[3]
   + 0.6931472042380503 lambda[4] + 0.6931472042380503 lambda[5]
   + 0.6931472042380503 lambda[6] + 0.6931472042380503 lambda[7]
   + 0.6931472042380503 lambda[8] + 0.6931472042380503 lambda[9]
   + 0.6931472042380503 lambda[10] + 0.6931472042380503 lambda[11]
   + 0.6931472042380503 lambda[12] + 0.6931472042380503 lambda[13]
   + 0.6931472042380503 lambda[14] + 0.6931472042380503 lambda[15]
   + 0.6931472042380503 lambda[16] + 0.6931472042380503 lambda[17]
   + 0.6931472042380503 lambda[18] + 0.6931472042380503 lambda[19]
   + 0.6931472042380503 lambda[20] + 0.6931472042380503 lambda[21]
   + 0.6931472042380503 lambda[22] + 0.6931472042380503 lambda[23]
   + 0.6931472042380503 lambda[24] + 4.05688274522056 lambda[25]
   + 0.0510443326193033 lambda[26] - phi[0] = 0
Bounds
End
