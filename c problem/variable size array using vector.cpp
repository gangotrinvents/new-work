#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;

int main() {
int a1,q,values; //length of array and query
cin>>a1>>q;
vector<int> v[a1];
int l;  //length of array's array

for(int i=0;i<a1;i++)
{
    cin>>l;
for(int j=0;j<l;j++)
{

cin>>values;
v[i].push_back(values);
}
}
int va;
vector<int> tofind[q];
for(int i=0;i<q;i++)
{
    for(int j=0;j<2;j++)
    {
      cin>>va;
     tofind[i].push_back(va);
    }

}

for(int i=0;i<q;i++)
{
    cout<<v[tofind[i].at(0)].at(tofind[i].at(1))<<"\n";
}
 return 0;
}

