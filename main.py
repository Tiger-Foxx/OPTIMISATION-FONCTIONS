import numpy as np

#a=np.array([1, 2, 3])
#b=np.array([-1 ,0,2])
 

#def multiplier(a,b,x1,x2,y1,y2):
#    print(np.multiply(a.reshape(x1,x2),b.reshape(y1,y2)))

#multiplier(a,b,3,1,1,3)

a=[[1],[2],[3]]
b=[[-1,0,2]]
def afficher(A):
    for i in range(len(A)):
        print(A[i])
def multiplier2(a,b):
    
    x=0
    for i in range(len(a)):
        for j in range(len(b[0])):
          x=0
          for k in range(len(b)):
           x = x + a[i][k]*b[k][j]
          print(x,end=" ")
        print("\n")

def multiplier3(a,b):
    c=[[0 for i in range(len(b[0]))] for i in range(len(a))]
    
    x=0
    for i in range(len(a)):
        for j in range(len(b[0])):
          x=0
          for k in range(len(b)):
           x = x + a[i][k]*b[k][j]
           #print(x,end=" ")
          c[i][j]=x
    
    return c
        #print("\n")

def trace(a):
    sum=0
    for i in range(len(a)):
        for j in range(len(a[0])):
            if i==j:
                sum=sum+a[i][j]
    print(sum)
#trace(a)

def maxx(a,b,c):
    if a>=b:
        if a>=c:
            print(f"le max est {a}")
            pass
    elif(b>=c):
        print(f"le max est {b}")
        pass
    else:
        print(f"le max est {c}")
        pass
    
def max():
    max=input("le premier : ")
    for i in range(2):
      x=input("le second : ")
      if max<x:
          max=x
    print(f"max est {max}")
    
    

def maxMatrice(a):
    x=len(a[0])
    y=len(a)
    max=a[0][0]
    for i in range(y):
        for j in range(x):
            if a[i][j]>max:
                max=a[i][j]
    print(f"le max de {a} est {max}")

def multMat(a,t):
    x=len(a[0])
    y=len(a)
    max=a[0][0]
    for i in range(y):
        for j in range(x):
            a[i][j]=t*a[i][j]
#maxMatrice(a)

def det2X2(a):
    return a[0][0]*a[1][1]-a[1][2]*a[1][0]




    