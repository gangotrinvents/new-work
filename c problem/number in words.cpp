#include<iostream>
using namespace std;
void main()
{
int n;
string str1="one two three four five six seven eight nine ten eleven tweleve thirteen fourteen fifteen sixteen seventeen eighteen ninteen";
string str2="twenty thirty fourty fifty sixty sevevnty eighty ninty hundred";
string str3="thousand lakh";
cout<<"enter number to convert in words";
cin>>n;
int temp=n,count=0;
while(temp>0)
    {
    temp=temp%10;
    count=count+1;
    }

if(count=3)
    {
    temp=temp%100;
    if(temp<20)
        {

        }
    }
}
