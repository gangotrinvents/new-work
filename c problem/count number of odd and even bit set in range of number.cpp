#include <stdio.h>
#include<iostream>
using namespace std;

int main()
{
    int n,even=0,odd=0,ones=0;
    cout<<"enter number";

    cin>>n;
    int t=n;
     int count=0,cnt=0;
    while(t){
        ones+=t&1;
        t>>=1;
    }
    cout<<ones<<"ones";
    if(ones>1){

    while(n){
        count=count+n&10;
        if(count != 0)
        {
            break;
        }
        n=n>>4;
        cout<<count;
    }
    if(count == 0)
    {
        odd+=1;
        cout<<odd<<"odd";
    }
    while(n){
        cnt=cnt+n&5;
        if(cnt != 0)
        {
            break;
        }
        n=n>>4;
        cout<<cnt;
    }
    if(cnt == 0)
    {
        even+=1;
        cout<<even<<"even";
    }
    }


    return 0;
}
