\ Model iteration_7
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  vars[8] + vars[9]
Subject To
 R0: - vars[0] + vars[1] <= 10000
 R1: vars[0] - vars[1] <= -0.001
 R2: - vars[0] + vars[2] <= 10000
 R3: vars[0] - vars[2] <= -0.001
 R4: - vars[2] + vars[3] <= 30
 R5: vars[2] - vars[3] <= -30
 R6: - vars[3] + vars[4] <= 10000
 R7: vars[3] - vars[4] <= -0.001
 R8: - vars[2] + vars[4] <= 10000
 R9: vars[2] - vars[4] <= -0.001
 R10: - vars[1] + vars[6] <= 10000
 R11: vars[1] - vars[6] <= -0.001
 R12: - vars[4] + vars[5] <= 10
 R13: vars[4] - vars[5] <= -10
 R14: - vars[2] + vars[5] <= 10000
 R15: vars[2] - vars[5] <= -0.001
 R16: - vars[0] + vars[7] - vars[9] <= 60.001
 R17: vars[0] - vars[7] - vars[8] <= 0
 R18: - vars[5] + vars[7] <= 10000
 R19: vars[5] - vars[7] <= -0.001
 z0: 0.1666666666666667 vars[1] - 0.1666666666666667 vars[6] - 3 lambda[0]
   - 3 lambda[2] - 3 lambda[3] - 3 lambda[4] - 3 lambda[5] - 3 lambda[6]
   >= -1671.666666666667
 z1: - 0.1666666666666667 vars[1] + 0.1666666666666667 vars[6]
   - 3 lambda[0] - 3 lambda[1] - 3 lambda[3] - 3 lambda[4] - 3 lambda[5]
   - 3 lambda[6] >= -4.999833333333333
 z2: 0.1666666666666667 vars[6] - 0.1666666666666667 vars[7] - 3 lambda[0]
   - 3 lambda[1] - 3 lambda[2] - 3 lambda[4] - 3 lambda[5] - 3 lambda[6]
   >= -1671.666666666667
 z3: - 0.1666666666666667 vars[1] + 0.1666666666666667 vars[6]
   - 3 lambda[0] - 3 lambda[1] - 3 lambda[2] - 3 lambda[3] - 3 lambda[5]
   - 3 lambda[6] >= 5.000166666666667
 z4: 0.1666666666666667 vars[1] - 0.1666666666666667 vars[6] - 3 lambda[0]
   - 3 lambda[1] - 3 lambda[2] - 3 lambda[3] - 3 lambda[4] - 3 lambda[6]
   >= -1661.666666666667
 z5: - 0.1666666666666667 vars[6] + 0.1666666666666667 vars[7]
   - 3 lambda[0] - 3 lambda[1] - 3 lambda[2] - 3 lambda[3] - 3 lambda[4]
   - 3 lambda[5] >= 5.000166666666667
 cc: 0.00540689417095195 lambda[0] + 0.6985540747308973 lambda[1]
   + 0.6985540747308973 lambda[2] + 0.6985540747308973 lambda[3]
   + 0.6985540747308973 lambda[4] + 0.6985540747308973 lambda[5]
   + 0.6985540747308973 lambda[6] <= 0.2231435513142097
 R27: vars[0] = 0
 R28: lambda[0] + lambda[1] + lambda[2] + lambda[3] + lambda[4] + lambda[5]
   + lambda[6] = 1
Bounds
End
