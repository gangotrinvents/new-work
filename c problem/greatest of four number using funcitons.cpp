#include <iostream>
#include <cstdio>
using namespace std;
int max_of_two(int q, int p)
{
    int a =q;
    int b = p;
    if(a>b)
    {
        return a;
    }
    else{
    return b;
    }
}
int main() {
    int a, b, c, d;
    scanf("%d %d %d %d", &a, &b, &c, &d);
    int x = max_of_two(a, b);
    int y = max_of_two(c,d);
    int ans = max_of_two(x,y);
    printf("%d", ans);

    return 0;
}

