#include<iostream>
#include<string.h>
using namespace std;
main()
{
    string str=" absd sdbsdfb absdf";
    int l=0,a[26]={0},l2=0;
    for(l=0;str[l]!='\0';l++);
    cout<<"lenth of string :"<<l;
    for(int i=0;i<l;i++)
    {
        if(str[i]>='a'&& str[i]<='z')
        {
            a[str[i]-'a']=a[str[i]-'a']+1;
        }
    }
    for(int i=0;i<26;i++)
    {
        if(a[i]>0)
        {
            l2=l2+1;
        }
    }
    cout<<"\n"<<"total diff values are :"<<l2<<"\n";
    char sub[l2];
    int j=0;
    for(int i=0;i<26;i++)
    {
        if(a[i]>0)
        {
            sub[j]=char('a'+i);
            cout<<sub[j];
            j=j+1;
        }
    }
}
