#include<iostream>
#include<cstdio>
#include<string>
#include<vector>

// bool Stringcheck(String);
bool Stringcheck(char *str);
{
    int temp,n;
    for(int n=0;str[n]!='\0',n++);
    cout<<n;
    int diff[n-1];
    for(int i=0;i<n-1;i++)
    {
        diff[i]=abs(int(str[i+1])-int(str[i]));
    }
    int newarr[n];
    for(int i=0;i<n;i++)
    {
        newarr[i]=str[n-i];
    }
    for(int i=0;i<n-1;i++)
    {
        if(abs(int(str[i+1])-int(str[i]))!=diff[i])
            return false;
        else{ return true;}
    }
}
int main()
{
    char str[100];
    cout<<"enter string : ";
    cin>>str;
    char str2[100];
    strcpy(str2,str);
    cout<<str<<"is"<<(Stringcheck(str2)?"acceptable":"not acceptable";

}
//bool Stringcheck(string org)
//{
//string rev;
//rev = org;
//reverse(rev.begin(),rev.end());
//}
