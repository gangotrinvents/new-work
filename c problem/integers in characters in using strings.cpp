#include <bits/stdc++.h>
using namespace std;
int main()
{
    string a="one two three four five six seven eight nine";
    int number;
    cin>>number;
    int count=0;
    if(number<=9 && number>=1)
    {
    cout<<"enter";
        for(int i=0;a[i] !='\0';i++)
        {
            if(a[i]==' ')
            {
                count=count+1;
            }
            if(count == number-1)
            {
                cout<<a[i];
            }
        }
    }
    else{ cout<<"Greater than 9";}
return 0;
}
