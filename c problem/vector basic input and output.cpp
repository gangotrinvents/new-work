#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>
using namespace std;
int main() {
  vector<int> v;
  int n;
  int value;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> value;
    v.push_back(value);
  }
  for (int i = 0; i < v.size(); i++) {
    cout << v.at(v.size() - i - 1)<<" ";
  }
  return 0;
}
