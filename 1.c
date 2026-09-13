#include <stddef.h>
double clamp_hi(double v1, double v2) {
  double v3 = 0.0e+00;
  double v4 = 1.00000000000000000e+00;
  size_t v5 = 0;
  double v6[1];
  double v7[1];
  double v8[1];
  double v9[1];
  v6[v5] = v1;
  v7[v5] = v2;
  double v10 = v6[v5];
  v8[v5] = v10;
  double v11 = v6[v5];
  double v12 = v7[v5];
  bool v13 = v11 > v12;
  bool v14 = v11 == v11;
  bool v15 = v12 == v12;
  bool v16 = v14 && v15;
  bool v17 = v16 && v13;
  double v18 = v17 ? v4 : v3;
  v9[v5] = v18;
  double v19 = v9[v5];
  bool v20 = v19 != v3;
  bool v21 = v19 != v19;
  bool v22 = v3 != v3;
  bool v23 = v21 || v22;
  bool v24 = v23 || v20;
  if (v24) {
    double v25 = v7[v5];
    v8[v5] = v25;
  }
  double v26 = v8[v5];
  return v26;
}

#include <stdio.h>

int main() {
  double res = clamp_hi(1, 2);
  printf("res: %f\n", res);
  return 0;
}
