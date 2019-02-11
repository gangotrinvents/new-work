#include <iostream>
#include <cstdio>
using namespace std;
int main()
{
    string a="one two three four five six seven eight nine";
    int numbere[2];
    for(int i=0;i<2;i++)
    {
    cin>>numbere[i];
    }
    int count=0,number;
    number=numbere[0];
    if(number<=9 && number>=1 )
    {
    for(int i=0;a[i] !='\0';i++)
        {
          if(count >= number-1 )
            {
                cout<<a[i];
            }
            if(a[i]==' ')
            {
                count=count+1;
            }
            if(count > number-1 && a[i]==' ')
            {
                cout<<"\n";
            }
            if(count == numbere[1])
            {
                break;
            }
        }
        cout<<"\n";
    }
    if(numbere[0] >= 10)
    {
     for(int i=numbere[0];i<=numbere[1];i++)
     {
         if(i%2==0)
        {
            cout<<"even \n";
        }
        else{
            cout<<"odd \n";
        }
     }
    }
    if(numbere[0] <=10 && numbere[1]>=10)
    {
        for(int i=10;i<=numbere[1];i++)
     {
         if(i%2==0)
        {
            cout<<"even \n";
        }
        else{
            cout<<"odd \n";
        }
     }
    }
return 0;
}

