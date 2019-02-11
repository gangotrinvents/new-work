#include<iostream>
using namespace std;
int main()
{
char str[100]="hello brother";
int freq[26]={0},j,i=0;
while(str[i] != '\0')
{
if(str[i]>='a' && str[i]<='z')
{
 j = str[i] - 'a';
 freq=freq[j]+1;

}
i++
}
  cout<<"Frequency of all alphabets in the string is:"<<endl;
   for (i = 0; i < 26; i++)
      cout<< char(i + 'a')<<" : "<< alphabet[i]<< endl;
   return 0;
}
