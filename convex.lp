\ Model iteration_12
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
   + 3 lambda[11] <= 1671.666666666667
 z1: 0.1666666666666667 vars[1] - 0.1666666666666667 vars[6] + 3 lambda[0]
   + 3 lambda[1] + 3 lambda[3] + 3 lambda[4] + 3 lambda[5] + 3 lambda[6]
   + 0.5085550764877833 lambda[7] + 1.471029661752156 lambda[8]
   + 1.001803417672902 lambda[9] + 1.225183279638509 lambda[10]
   + 1.376268133477439 lambda[11] <= -5.000166666666667
 z2: - 0.1666666666666667 vars[1] + 0.1666666666666667 vars[6]
   + 3 lambda[0] + 3 lambda[1] + 3 lambda[2] + 3 lambda[4] + 3 lambda[5]
   + 3 lambda[6] + 3 lambda[7] + 3 lambda[8] + 3 lambda[9] + 3 lambda[10]
   + 3 lambda[11] <= 1661.666666666667
 z3: 0.1666666666666667 vars[1] - 0.1666666666666667 vars[6] + 3 lambda[0]
   + 3 lambda[1] + 3 lambda[2] + 3 lambda[3] + 3 lambda[5] + 3 lambda[6]
   + 3 lambda[7] + 3 lambda[8] + 3 lambda[9] + 3 lambda[10] + 3 lambda[11]
   <= 4.999833333333333
 z4: - 0.1666666666666667 vars[6] + 0.1666666666666667 vars[7]
   + 3 lambda[0] + 3 lambda[1] + 3 lambda[2] + 3 lambda[3] + 3 lambda[4]
   + 3 lambda[6] + 3 lambda[7] + 3 lambda[8] + 3 lambda[9] + 3 lambda[10]
   + 3 lambda[11] <= 1671.666666666667
 z5: 0.1666666666666667 vars[6] - 0.1666666666666667 vars[7] + 3 lambda[0]
   + 3 lambda[1] + 3 lambda[2] + 3 lambda[3] + 3 lambda[4] + 3 lambda[5]
   + 1.071632200684582 lambda[7] + 1.499076602353606 lambda[8]
   + 0.9997005055156135 lambda[9] + 1.222760355426867 lambda[10]
   + 1.374261257531198 lambda[11] <= -5.000166666666667
 cc: 0.00540689417095195 lambda[0] + 0.6985540747308973 lambda[1]
   + 0.6985540747308973 lambda[2] + 0.6985540747308973 lambda[3]
   + 0.6985540747308973 lambda[4] + 0.6985540747308973 lambda[5]
   + 0.6985540747308973 lambda[6] + 0.5212137235136484 lambda[7]
   + 0.1454338091889734 lambda[8] + 0.3482864372787983 lambda[9]
   + 0.2371876630394646 lambda[10] + 0.1795752922891036 lambda[11]
   <= 0.2231435513142097
 x0: vars[0] = 0
 sum_lam: lambda[0] + lambda[1] + lambda[2] + lambda[3] + lambda[4]
   + lambda[5] + lambda[6] + lambda[7] + lambda[8] + lambda[9] + lambda[10]
   + lambda[11] = 1
Bounds
End
