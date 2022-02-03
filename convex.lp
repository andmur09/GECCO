\ Model iteration_28
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  vars[8] + vars[9]
Subject To
 cont[0]: - vars[0] + vars[1] <= 10000
 cont[1]: vars[0] - vars[1] <= -0.001
 cont[2]: - vars[0] + vars[2] <= 10000
 cont[3]: vars[0] - vars[2] <= -0.001
 cont[4]: - vars[2] + vars[3] <= 30
 cont[5]: vars[2] - vars[3] <= -30
 cont[6]: - vars[3] + vars[4] <= 10000
 cont[7]: vars[3] - vars[4] <= -0.001
 cont[8]: - vars[2] + vars[4] <= 10000
 cont[9]: vars[2] - vars[4] <= -0.001
 cont[10]: - vars[1] + vars[6] <= 10000
 cont[11]: vars[1] - vars[6] <= -0.001
 cont[12]: - vars[4] + vars[5] <= 10
 cont[13]: vars[4] - vars[5] <= -10
 cont[14]: - vars[2] + vars[5] <= 10000
 cont[15]: vars[2] - vars[5] <= -0.001
 cont[16]: - vars[0] + vars[7] - vars[9] <= 60.001
 cont[17]: vars[0] - vars[7] - vars[8] <= 0
 cont[18]: - vars[5] + vars[7] <= 10000
 cont[19]: vars[5] - vars[7] <= -0.001
 z0: - 0.1666666666666667 vars[1] + 0.1666666666666667 vars[6]
   + 3 lambda[0] + 3 lambda[2] + 3 lambda[3] + 3 lambda[4] + 3 lambda[5]
   + 3 lambda[6] + 3 lambda[7] + 3 lambda[8] + 3 lambda[9] + 3 lambda[10]
   + 3 lambda[11] + 3 lambda[12] + 3 lambda[13] + 3 lambda[14]
   + 3 lambda[15] + 3 lambda[16] + 3 lambda[17] + 3 lambda[18]
   + 3 lambda[19] + 3 lambda[20] + 3 lambda[21] + 3 lambda[22]
   + 3 lambda[23] + 3 lambda[24] + 3 lambda[25] + 3 lambda[26]
   + 3 lambda[27] <= 1671.666666666667
 z1: 0.1666666666666667 vars[1] - 0.1666666666666667 vars[6] + 3 lambda[0]
   + 3 lambda[1] + 3 lambda[3] + 3 lambda[4] + 3 lambda[5] + 3 lambda[6]
   + 2.057617213559045 lambda[7] + 1.456854495597499 lambda[8]
   + 1.328122280018805 lambda[9] + 1.210765245268312 lambda[10]
   + 1.103778244823119 lambda[11] + 1.006244950047949 lambda[12]
   + 0.9292983463800299 lambda[13] + 1.00112952216189 lambda[14]
   + 1.10419959566 lambda[15] + 1.097846770277875 lambda[16]
   + 1.091724694478739 lambda[17] + 1.085824986894171 lambda[18]
   + 1.080139570586794 lambda[19] + 1.074660661992621 lambda[20]
   + 1.069380760265046 lambda[21] + 1.064292637005883 lambda[22]
   + 1.059389326369393 lambda[23] + 0.6507034475460007 lambda[24]
   + 0.5862298459326633 lambda[25] + 1.30662583 lambda[26]
   + 2.00714409 lambda[27] <= -5.000166666666667
 z2: - 0.1666666666666667 vars[1] + 0.1666666666666667 vars[6]
   + 3 lambda[0] + 3 lambda[1] + 3 lambda[2] + 3 lambda[4] + 3 lambda[5]
   + 3 lambda[6] + 3 lambda[7] + 3 lambda[8] + 3 lambda[9] + 3 lambda[10]
   + 3 lambda[11] + 3 lambda[12] + 3 lambda[13] + 3 lambda[14]
   + 3 lambda[15] + 3 lambda[16] + 3 lambda[17] + 3 lambda[18]
   + 3 lambda[19] + 3 lambda[20] + 3 lambda[21] + 3 lambda[22]
   + 3 lambda[23] + 3 lambda[24] + 3 lambda[25] + 3 lambda[26]
   + 3 lambda[27] <= 1661.666666666667
 z3: 0.1666666666666667 vars[1] - 0.1666666666666667 vars[6] + 3 lambda[0]
   + 3 lambda[1] + 3 lambda[2] + 3 lambda[3] + 3 lambda[5] + 3 lambda[6]
   + 3 lambda[7] + 3 lambda[8] + 3 lambda[9] + 3 lambda[10] + 3 lambda[11]
   + 3 lambda[12] + 3 lambda[13] + 3 lambda[14] + 3 lambda[15]
   + 3 lambda[16] + 3 lambda[17] + 3 lambda[18] + 3 lambda[19]
   + 3 lambda[20] + 3 lambda[21] + 3 lambda[22] + 3 lambda[23]
   + 3 lambda[24] + 3 lambda[25] + 3 lambda[26] + 3 lambda[27]
   <= 4.999833333333333
 z4: - 0.1666666666666667 vars[6] + 0.1666666666666667 vars[7]
   + 3 lambda[0] + 3 lambda[1] + 3 lambda[2] + 3 lambda[3] + 3 lambda[4]
   + 3 lambda[6] + 3 lambda[7] + 3 lambda[8] + 3 lambda[9] + 3 lambda[10]
   + 3 lambda[11] + 3 lambda[12] + 3 lambda[13] + 3 lambda[14]
   + 3 lambda[15] + 3 lambda[16] + 3 lambda[17] + 3 lambda[18]
   + 3 lambda[19] + 3 lambda[20] + 3 lambda[21] + 3 lambda[22]
   + 3 lambda[23] + 3 lambda[24] + 3 lambda[25] + 3 lambda[26]
   + 3 lambda[27] <= 1671.666666666667
 z5: 0.1666666666666667 vars[6] - 0.1666666666666667 vars[7] + 3 lambda[0]
   + 3 lambda[1] + 3 lambda[2] + 3 lambda[3] + 3 lambda[4] + 3 lambda[5]
   + 3 lambda[7] + 3 lambda[8] + 3 lambda[9] + 3 lambda[10] + 3 lambda[11]
   + 3 lambda[12] + 1.333826296985687 lambda[13]
   + 2.889232389512583 lambda[14] + 2.243462437859906 lambda[15]
   + 2.210422310427802 lambda[16] + 2.178582278255413 lambda[17]
   + 2.14789875105589 lambda[18] + 2.11832972184428 lambda[19]
   + 2.089834709428284 lambda[20] + 2.062374702987862 lambda[21]
   + 2.035912108667858 lambda[22] + 2.010410698110492 lambda[23]
   + 1.433365005126602 lambda[24] + 0.6645504481759644 lambda[25]
   + 1.07092105 lambda[26] + 1.07420522 lambda[27] <= -5.000166666666667
 cc: 0.00540689417095195 lambda[0] + 0.6985540747308973 lambda[1]
   + 0.6985540747308973 lambda[2] + 0.6985540747308973 lambda[3]
   + 0.6985540747308973 lambda[4] + 0.6985540747308973 lambda[5]
   + 0.6985540747308973 lambda[6] + 0.0270986346403633 lambda[7]
   + 0.1770629675384519 lambda[8] + 0.1770629675384519 lambda[9]
   + 0.1770629675384519 lambda[10] + 0.1770629675384519 lambda[11]
   + 0.1770629675384519 lambda[12] + 0.8702101480983973 lambda[13]
   + 0.1987547080078633 lambda[14] + 0.1987547080078633 lambda[15]
   + 0.1987547080078633 lambda[16] + 0.1987547080078633 lambda[17]
   + 0.1987547080078633 lambda[18] + 0.1987547080078633 lambda[19]
   + 0.1987547080078633 lambda[20] + 0.1987547080078633 lambda[21]
   + 0.1987547080078633 lambda[22] + 0.1987547080078633 lambda[23]
   + 0.8702101480983973 lambda[24] + 1.391701255290843 lambda[25]
   + 0.348719040905952 lambda[26] + 0.1987547080078633 lambda[27]
   <= 0.2231435513142097
 x0: vars[0] = 0
 sum_lam: lambda[0] + lambda[1] + lambda[2] + lambda[3] + lambda[4]
   + lambda[5] + lambda[6] + lambda[7] + lambda[8] + lambda[9] + lambda[10]
   + lambda[11] + lambda[12] + lambda[13] + lambda[14] + lambda[15]
   + lambda[16] + lambda[17] + lambda[18] + lambda[19] + lambda[20]
   + lambda[21] + lambda[22] + lambda[23] + lambda[24] + lambda[25]
   + lambda[26] + lambda[27] = 1
Bounds
End
