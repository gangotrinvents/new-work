#include <stdio.h>
#include<iostream>
using namespace std;

int main()
{
    int n,count=0;
    cout<<"enter number";
    cin>>n;
    while(n){
        count=count+(n&1);
        n=n>>1;
    }
    cout<<count;
    return 0;
}
