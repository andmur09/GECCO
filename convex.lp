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
   + 1.265139904731552 lambda[7] - 2.983144129154117 lambda[8]
   - 1.031995791952785 lambda[9] - 0.3507870512622664 lambda[10]
   + 1.025554683595239 lambda[11] <= -5.000166666666667
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
   + 2.012675686750506 lambda[7] - 2.609044106781853 lambda[8]
   - 0.8400603609949602 lambda[9] - 0.6419627958822565 lambda[10]
   + 1.202589669888633 lambda[11] <= -5.000166666666667
 cc: 0.00540689417095195 lambda[0] + 0.6985540747308973 lambda[1]
   + 0.6985540747308973 lambda[2] + 0.6985540747308973 lambda[3]
   + 0.6985540747308973 lambda[4] + 0.6985540747308973 lambda[5]
   + 0.6985540747308973 lambda[6] + 0.1987547080078633 lambda[7]
   + 7.68870685266841 lambda[8] + 2.54541704819303 lambda[9]
   + 1.391701255290843 lambda[10] + 0.348719040905952 lambda[11]
   <= 0.2231435513142097
 x0: vars[0] = 0
 sum_lam: lambda[0] + lambda[1] + lambda[2] + lambda[3] + lambda[4]
   + lambda[5] + lambda[6] + lambda[7] + lambda[8] + lambda[9] + lambda[10]
   + lambda[11] = 1
Bounds
End
