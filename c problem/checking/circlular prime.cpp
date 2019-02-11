using namespace std;
int main()
{
    int a,temp,l,mod,rem,s,m;
    cout<<"enter number to work";
    cin>>a;
    m=a;
    cout<<"number =="<<m;
    while(m!=0)
    {
        m=m%10;
        l+=1;
    }
    cout<<l;
    for(int i=0;i<l;i++)
    {
        mod=a/10;
        rem=a%10;
        s=pow(10,l-1);
        temp=rem*pow(10,l-1)+mod;
        cout<<"num is : "<<s;
        a=temp;
    }
}
