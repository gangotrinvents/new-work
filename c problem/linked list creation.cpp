#include<iostream>
using namespace std;
struct node
{
int data;
node *next;
};
class linkedx
{
node *head=NULL,*tail=NULL;
public:
void createNode(int x)
    {
    node *temp= new node;
    temp->data=x;
    temp->next=NULL;
    if(head==NULL)
        {
        head=temp;
        tail=temp;
        temp=NULL;
        }
    else
        {
        tail->next=temp;
        tail=temp;
        }
    }
void display()
    {
    node *temp=head;
    while(temp!=NULL)
        {
        cout<<"linked list data :"<<temp->data;
        if (temp->next!=NULL)
            {
            cout<<"==>";
            }
        temp=temp->next;
        }
    }
void insertonhead(int z)
    {
    node *temp=new node;
    temp->data=z;
    temp->next=head;
    head=temp;
    temp=NULL;
    }
void shift(int y)
    {
    cout<<"y is:"<<y;
    while(y!=0)
        {
        node *temp=head;
        node *temp1= new node;
        cout<<"head   :  "<<head->data<<"\n";
        while(temp!=NULL)
            {
            if(temp->next==NULL)
                {
                cout<<"\n value : "<<temp->data;
                temp1=temp;
                temp1->next=head;
                temp=NULL;
                head=temp1;
                cout<<"head "<<head->data;
                cout<<" \n head next"<<head->next;
                }
            else {
                    temp=temp->next;
                }
            }
        cout<<y;
        y=y-1;
        cout<<"nsd : "<<y;
        }
    }
};
int main()
{
    linkedx a;
    a.createNode(42);
    a.createNode(23);
    a.insertonhead(100);
    a.display();
    a.shift(2);
    a.display();

}
