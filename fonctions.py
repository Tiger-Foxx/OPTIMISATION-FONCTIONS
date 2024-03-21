# 2.2

import copy
import random
from re import X

from sympy import N

from main import afficher, multiplier2, multiplier3


def multMatVal(a,X):
    for i in range(len(X)):
        for j in range(len(X[0])):
            X[i][j]=a*X[i][j]
    return X
def transpose(X):
    try :
        Y=[[X[i][j] for i in range(len(X))] for j in range(len(X[0]))]
    except :
        Y=[[X[i]] for i in range(len(X))]
        
    return Y

def add(X,Y):
    
    try:
        if len(X[0])>1 :
           Z= [[0 for i in range(len(X[0]))] for j in range(len(X))]
           for i in range(len(X)):
            for j in range(len(X[0])):
                Z[i][j]=X[i][j]+Y[i][j]
                return Z
        else:
            Z=[X[i]+Y[i] for i in range( len(X))]
            return Z
    except Exception as e:
        Z=[X[i]+Y[i] for i in range( len(X))]
        return Z
    

def FINDLARGESTELEMENT(liste):
    """Retourne le plus grand élément de la liste fournie."""
    plus_grand = liste[0]
    for element in liste[1:]:
        if element > plus_grand:
            plus_grand = element
    print("Le plus grand élément est : ", plus_grand)

# FINDLARGESTELEMENT([3, 67, 99, 23, 45])


# 2.3

def FINDLARGESTELEMENT2(liste):
    """Retourne le plus grand élément de la liste fournie."""
    if not liste:
        print("La liste est vide.")
        return
    plus_grand = liste[0]
    i = 1
    while i < len(liste):
        if liste[i] > plus_grand:
            plus_grand = liste[i]
        i += 1
    print("Le plus grand élément est :", plus_grand)

# FINDLARGESTELEMENT([3, 67, 99, 23, 45])

# 2.4

def ISPRIME(n):
    """Vérifie si un nombre est premier."""
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1): #car un de ses diviseurs est forcement plus petit ou egal a sa moitie
        if n % i == 0:
            return False
    return True

# 2.5

def LARGEPRIME(n):
    """Trouve le plus petit entier premier supérieur à n."""
    x=n
    n += 1
    while not ISPRIME(n):
        n += 1
    
    print("Le plus petit entier premier supérieur à", x, "est :", n)
    return n

# Pour tester les fonctions, décommentez les lignes suivantes et exécutez-les.
# print("Est premier :", ISPRIME(29))
#LARGEPRIME(29)


# 2.6

def CHERCHEELEMENT(liste, element):
    """Cherche un élément dans une liste donnée."""
    for j,i in enumerate(liste) :
        if i == element:
            print(f"Élément trouvé {element} position {j}")
            return
    print("Élément non trouvé.")

#a=[random.randrange(1,80,1) for _ in range(80)]

#CHERCHEELEMENT(a,9)

# 2.7

def LARGESTCOMMONDIVISOR(a, b):
    """Calcule le PGCD de deux nombres a et b."""
    while b:
        a, b = b, a % b
    return a

# Pour tester la fonction PGCD, décommentez la ligne suivante et exécutez-la.
#print("Le PGCD est :", LARGESTCOMMONDIVISOR(48, 18))


# 2.8

def LARGESTCOMMONDIVISOR2(a, b):
    while b:
        a, b = b, a % b
    return a

def LOWESTCOMMONMULTIPLE(a, b):
    """Calcule le plus petit multiple commun (PPCM) de deux nombres a et b."""
    return abs(a*b) // LARGESTCOMMONDIVISOR(a, b)

# 2.9

def LARGESTCOMMONDIVISORREC(a, b):
    """Calcule le PGCD de manière récursive."""
    if b == 0:
        return a
    else:
        return LARGESTCOMMONDIVISORREC(b, a % b)

# 2.10

def LOWESTCOMMONMULTIPLEREC(a, b):
    """Calcule le PPCM de deux nombres a et b de manière récursive."""
    return abs(a*b) // LARGESTCOMMONDIVISORREC(a, b)

# 2.11

def P_COMBINATION2(n, p):
    """Trouve une p-combinaison de 1, 2, 3, ..., n."""
    from math import factorial
    if p > n:
        return "p ne peut pas être plus grand que n."
    return factorial(n) // (factorial(p) * factorial(n - p))


# 2.11

from math import comb, sin

def P_COMBINATION(n, p):
    """Génère toutes les p-combinaisons de 1, 2, 3, ..., n."""
    if p > n:
        return "p ne peut pas être plus grand que n."
    
    # Initialisation de la première combinaison
    s = [i for i in range(1, p + 1)]
    Out = [s.copy()]  # Stocke la première combinaison
    
    # Nombre total de combinaisons à générer
    total_combinations = comb(n, p)
    
    for _ in range(1, total_combinations):
        m = p - 1
        largestValue = n
        
        # Trouver l'indice à incrémenter
        while m >= 0 and s[m] == largestValue:
            m -= 1
            largestValue -= 1
        
        # Incrémenter cet indice
        s[m] += 1
        
        # Réinitialiser les indices suivants
        for j in range(m + 1, p):
            s[j] = s[j - 1] + 1
        
        Out.append(s.copy())  # Ajouter la nouvelle combinaison à la sortie
    
    return Out



# print("Le PPCM est :", LOWESTCOMMONMULTIPLE(48, 18))
# print("Le PGCD récursif est :", LARGESTCOMMONDIVISORREC(48, 18))
# print("Le PPCM récursif est :", LOWESTCOMMONMULTIPLEREC(48, 18))
#print("Une p-combinaison de 1, 2, 3, ..., n est :", P_COMBINATION(7, 3))

# 2.12

def N_PERMUTATIONS(n):
    """Génère toutes les permutations possibles des nombres de 1 à n."""
    if n == 0:
        return [[]]
    
    result = []
    for p in N_PERMUTATIONS(n - 1):
        for i in range(len(p) + 1):
            result.append(p[:i] + [n] + p[i:])
    return result

# Exemple d'utilisation
#print("Les permutations de 1 à n sont :", N_PERMUTATIONS(3))

# 2.13

def LARGESMALL(liste):
    """Trouve le plus grand et le plus petit élément d'une liste donnée."""
    if not liste:
        return None, None
    max_val = min_val = liste[0]
    for val in liste:
        if val > max_val:
            max_val = val
        elif val < min_val:
            min_val = val
    return max_val, min_val

# 2.14

def LINEARSEARCH(liste, element):
    """Retourne la position d'un élément dans une liste, ou 0 s'il n'est pas trouvé."""
    for i, val in enumerate(liste):
        if val == element:
            return i + 1  # Les positions commencent à 1
    return 0


# 2.15

def DECIMALTOBASEB(n, b):
    """Convertit un nombre décimal en base b."""
    if n == 0:
        return '0'
    digits = ''
    while n:
        digits += str(n % b)
        n //= b
    return digits[::-1]


#2.16

def BASEBTODECIMAL(s, b):
    """Convertit un nombre de la base b en décimal."""
    decimal = 0
    power = 0
    for digit in s[::-1]:
        decimal += int(digit) * (b ** power)
        power += 1
    return decimal

#2.17
def SELECTIONSORTREC(s):
    n=len(s)
    
    if n==1:
        return s
    maxIndex=0
    for i in range(1,n):
        if s[i]>s[maxIndex]:
            maxIndex=i
    s[maxIndex],s[n-1]=s[n-1],s[maxIndex]
    t= SELECTIONSORTREC(s[:n-1]) + [(s[n-1])]
    return t

#print(SELECTIONSORT([1, 65, 34, 23, 43, 66, 1, 34]))

# 2.18

def SELECTIONSORT(data):
    """Trie une liste en utilisant le tri par sélection."""
    for i in range(len(data)):
        min_index = i
        for j in range(i+1, len(data)):
            if data[j] < data[min_index]:
                min_index = j
        data[i], data[min_index] = data[min_index], data[i]
    return data

# 2.19

def BINARYSEARCH(s, key):
    """Recherche binaire récursive dans un tableau trié."""
    def recursive_search(s, key, left, right):
        if right >= left:
            mid = left + (right - left) // 2
            if s[mid] == key:
                return mid
            elif s[mid] > key:
                return recursive_search(s, key, left, mid-1)
            else:
                return recursive_search(s, key, mid+1, right)
        else:
            return -1
    return recursive_search(s, key, 0, len(s)-1)

# 2.20

def MERGESEQUENCE(S, C):
    """Fusionne deux tableaux triés en un seul tableau trié."""
    index_s, index_c = 0, 0
    merged_list = []
    while index_s < len(S) and index_c < len(C):
        if S[index_s] < C[index_c]:
            merged_list.append(S[index_s])
            index_s += 1
        else:
            merged_list.append(C[index_c])
            index_c += 1
    while index_s < len(S):
        merged_list.append(S[index_s])
        index_s += 1
    while index_c < len(C):
        merged_list.append(C[index_c])
        index_c += 1
    return merged_list

# 2.21

def MERGE(liste1, liste2):
    """Fusionne deux tableaux triés en un seul tableau trié."""
    resultat = []
    i, j = 0, 0
    while i < len(liste1) and j < len(liste2):
        if liste1[i] < liste2[j]:
            resultat.append(liste1[i])
            i += 1
        else:
            resultat.append(liste2[j])
            j += 1
    # Ajoute les éléments restants de liste1, s'il y en a
    while i < len(liste1):
        resultat.append(liste1[i])
        i += 1
    # Ajoute les éléments restants de liste2, s'il y en a
    while j < len(liste2):
        resultat.append(liste2[j])
        j += 1
    return resultat

#l1=range(5,17,2)
#l2=range(1,16,3)

#print(MERGE(l1,l2))

#3.2

def determinant_gauss(matrice):
    """Calcule le déterminant d'une matrice en utilisant la méthode de Gauss."""
    n = len(matrice)
    det = 1
    for i in range(n):
        # Trouver le pivot
        pivot = matrice[i][i]
        if pivot == 0:  # Si le pivot est 0, le déterminant est 0
            return 0
        det *= pivot
        # Élimination de Gauss pour les lignes suivantes
        for j in range(i+1, n):
            ratio = matrice[j][i] / pivot
            for k in range(n):
                matrice[j][k] -= ratio * matrice[i][k]
    return det

# Exemple d'utilisation
matrice_exemple = [
    [2, -1, 8],
    [-1, 2, -1],
    [11, -1, 2]
]
vecteur_exemple=[1,2,3]
#print("Le déterminant de la matrice est :", determinant_gauss(matrice_exemple))

def mult(A,x):
    return [a * x for a in A]
def multcol(A,x):
    return [[a[0] * x] for a in A]
        
def minus(A,B):
    return [a - b for a, b in zip(A, B)]

#A=[[random.randint(0,90) for i in range(1000)] for i in range(1000) ]
#B=[random.randint(0,90) for i in range(1000)]

def RESOUDRESYSTEMELINEAIRE(A,B):
    """mon algo resous un systeme lineaire

    Args:
        A (_matrice_): la matrice des coef
        B (_matrice_): la matrice des resultats
    """
    n=len(A)
    x=[0 for i in range(n)]
    Gp=0
    #je fais l'elimination
    for p in range(n-1):
        
        for k in range(p+1,n):
            
            #A[k]=A[k]-A[p]*(A[k][p]/A[p][p])
            if A[p][p]==0:
                print('pas de solution unique')
                return
            B[k]=B[k]-(A[k][p]/A[p][p])*B[p]
            A[k]=minus(A[k],mult(A[p],(A[k][p]/A[p][p])))
           #print((A[k][p]))
 
    x[n-1]=B[n-1]
    #print(B)
    for p in range(n-1,-1,-1):
        #print(p)
        
        for k in range(p+1,n):
            Gp=Gp+A[p][k]*x[k]
            
        x[p]=(B[p]-Gp)/A[p][p]
        Gp=0

    afficher(x)      
    #afficher(B)
            
#RESOUDRESYSTEMELINEAIRE(matrice_exemple, mult(vecteur_exemple,51*17))
#RESOUDRESYSTEMELINEAIRE(A, B)
def TROUVERMATRICEINVERSE(A):
    # je procede d'abord a la construction de la matrice identite
    tailleA=len(A)
    
    Id=[[0.0 for _ in range(tailleA)] for __ in range(tailleA)]
    for i in range(tailleA) :
        Id[i][i]=1
    
    # MAINTENANT LE CALCULE PEUT COMMENCER
    
    # on va triangulariser la matrice de gauche
    for k in range(tailleA-1):
        
        for n in range(k+1,tailleA):
            
            #A[k]=A[k]-A[p]*(A[k][p]/A[p][p])
            if A[k][k]==0:
                print('pas inversible')
                return
            
            Id[n]=minus(Id[n],mult(Id[k],(A[n][k]/A[k][k])))
            A[n]=minus(A[n],mult(A[k],(A[n][k]/A[k][k])))
            
           #print((A[k][p])) 
    # ON DIAGONALISE EN TRIANGULARISANT EN SENS INVERSE
    for k in range(tailleA-1,0,-1):
        
        for n in range(k-1,-1,-1):
            
            #A[k]=A[k]-A[p]*(A[k][p]/A[p][p])
            if A[k][k]==0:
                print('pas inversible')
                return
            Id[n]=minus(Id[n],mult(Id[k],(A[n][k]/A[k][k])))
            A[n]=minus(A[n],mult(A[k],(A[n][k]/A[k][k])))
           
           #print((A[k][p])) 
    # ON TRANSFORME EN L'IDENTITE
    for k in range(tailleA):
        A[k]=mult(A[k],1/(A[k][k]))
        Id[k]=mult(Id[k],1/(Id[k][k]))
    
    #afficher(Id)
    return Id


#TROUVERMATRICEINVERSE(matrice_exemple)
#multiplier2(TROUVERMATRICEINVERSE(matrice_exemple),matrice_exemple)

def remplir_matrice_tridiagonale(dim, val_diag_princ, val_diag_sup, val_diag_inf):
    """Crée une matrice tridiagonale de dimension 'dim'."""
    matrice = [[0 for _ in range(dim)] for _ in range(dim)]
    for i in range(dim):
        matrice[i][i] = val_diag_princ
        if i > 0:
            matrice[i][i-1] = val_diag_inf
        if i < dim - 1:
            matrice[i][i+1] = val_diag_sup
    for ligne in matrice:
        print(ligne)
    return matrice

# Exemple d'utilisation
dimension = 5  # Taille de la matrice
valeur_diagonale_principale = 4
valeur_diagonale_superieure = 1
valeur_diagonale_inferieure = -1

#matrice_tridiagonale = remplir_matrice_tridiagonale(dimension, valeur_diagonale_principale, valeur_diagonale_superieure, valeur_diagonale_inferieure)




def newton_optimisation(f, df, x0, tol=1e-5, max_iter=1000):
    """Trouve le point où la dérivée de f est nulle en utilisant la méthode de Newton.
    
    Paramètres :
    f : La fonction à optimiser.
    df : La dérivée de la fonction f.
    x0 : Estimation initiale du point critique.
    tol : La tolérance pour l'arrêt de l'algorithme.
    max_iter : Le nombre maximum d'itérations.
    
    Retourne :
    x : L'approximation du point critique.
    """
    x = x0
    for _ in range(max_iter):
        x_new = x - f(x) / df(x)
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x

# Exemple d'utilisation :
# f = lambda x: x**2 - 20  # La fonction à optimiser
# df = lambda x: 2*x       # La dérivée de la fonction
# x0 = 4.5                 # Estimation initiale
# point_critique = newton_optimisation(f, df, x0)
# print("Le point critique est :", point_critique)


def quasi_newton_optimisation(valeurs, x0, tol=1e-5, max_iter=1000):
    """Optimisation en utilisant une méthode de type Quasi-Newton.
    
    Paramètres :
    valeurs : Vecteur des valeurs de la fonction à optimiser.
    x0 : Estimation initiale du point critique.
    tol : La tolérance pour l'arrêt de l'algorithme.
    max_iter : Le nombre maximum d'itérations.
    
    Retourne :
    x : L'approximation du point critique.
    """
    x = x0
    n = len(valeurs)
    for _ in range(max_iter):
        # Calcul de la dérivée approximative
        df = (valeurs[min(n-1, x+1)] - valeurs[x]) / (1 if x+1 < n else 0)
        
        # Mise à jour de x en utilisant la dérivée approximative
        x_new = x - valeurs[x] / df
        
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x

# Exemple d'utilisation avec un vecteur de valeurs
valeurs_exemple = [1, 4, 9, 16, 25]  # Par exemple, les valeurs de la fonction f(x) = x^2
x0 = 2  # Estimation initiale
#point_critique = quasi_newton_optimisation(valeurs_exemple, x0)
#print("Le point critique est :", point_critique)



def levenberg_marquardt(valeurs, x0, lambda_, tol=1e-5, max_iter=1000):
    """Optimisation en utilisant la méthode de Levenberg-Marquardt.
    
    Paramètres :
    valeurs : Vecteur des valeurs de la fonction à optimiser.
    x0 : Estimation initiale du point critique.
    lambda_ : Paramètre de mise à l'échelle pour la méthode.
    tol : La tolérance pour l'arrêt de l'algorithme.
    max_iter : Le nombre maximum d'itérations.
    
    Retourne :
    x : L'approximation du point critique.
    """
    x = x0
    n = len(valeurs)
    for _ in range(max_iter):
        # Calcul de la dérivée approximative
        df = (valeurs[min(n-1, x+1)] - valeurs[x]) / (1 if x+1 < n else 0)
        
        # Mise à jour de x en utilisant une combinaison de descente de gradient et de méthode de Newton
        x_new = x - (1 / (df + lambda_)) * valeurs[x]
        
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x

# Exemple d'utilisation avec un vecteur de valeurs
valeurs_exemple = [1, 4, 9, 16, 25]  # Par exemple, les valeurs de la fonction f(x) = x^2
x0 = 2  # Estimation initiale
lambda_ = 0.01  # Paramètre de mise à l'échelle
#point_critique = levenberg_marquardt(valeurs_exemple, x0, lambda_)
#print("Le point critique est :", point_critique)



import numpy as np

def quasi_newton_multivariable(f, grad_f, x0, tol=1e-5, max_iter=100):
    """Optimisation en utilisant une méthode de type Quasi-Newton pour les fonctions multivariables.
    
    Paramètres :
    f : La fonction à optimiser.
    grad_f : Le gradient de la fonction f.
    x0 : Estimation initiale du point critique.
    tol : La tolérance pour l'arrêt de l'algorithme.
    max_iter : Le nombre maximum d'itérations.
    
    Retourne :
    x : L'approximation du point critique.
    """
    x = x0
    for _ in range(max_iter):
        grad = grad_f(x)
        # Mise à jour de x en utilisant l'inverse approché de la matrice hessienne
        x_new = x - np.linalg.inv(approx_hessian(grad)) @ grad
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new
    return x

def approx_hessian(grad):
    """Calcule une approximation de la matrice hessienne."""
    n = len(grad)
    hessian = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                hessian[i, j] = grad[i] * grad[j]
            else:
                hessian[i, j] = grad[i] ** 2
    return hessian

# Exemple d'utilisation :
# f = lambda x: x[0]**2 + x[1]**2  # La fonction à optimiser
# grad_f = lambda x: np.array([2*x[0], 2*x[1]])  # Le gradient de la fonction
# x0 = np.array([1.0, 1.0])  # Estimation initiale
# point_critique = quasi_newton_multivariable(f, grad_f, x0)
# print("Le point critique est :", point_critique)


import numpy as np

def newton_method_multivariable(f, grad_f, hess_f, x0, tol=1e-5, max_iter=100):
    """Optimisation en utilisant la méthode de Newton pour les fonctions multivariables.
    
    Paramètres :
    f : La fonction à optimiser.
    grad_f : Le gradient de la fonction f.
    hess_f : La matrice hessienne de la fonction f.
    x0 : Estimation initiale du point critique.
    tol : La tolérance pour l'arrêt de l'algorithme.
    max_iter : Le nombre maximum d'itérations.
    
    Retourne :
    x : L'approximation du point critique.
    """
    x = x0
    for _ in range(max_iter):
        grad = grad_f(x)
        hess = hess_f(x)
        # Mise à jour de x en utilisant l'inverse de la matrice hessienne
        x_new = x - np.linalg.inv(hess) @ grad
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new
    return x

# Exemple d'utilisation :
# f = lambda x: x[0]**2 + x[1]**2  # La fonction à optimiser
# grad_f = lambda x: np.array([2*x[0], 2*x[1]])  # Le gradient de la fonction
# hess_f = lambda x: np.array([[2, 0], [0, 2]])  # La matrice hessienne de la fonction
# x0 = np.array([1.0, 1.0])  # Estimation initiale
# point_critique = newton_method_multivariable(f, grad_f, hess_f, x0)
# print("Le point critique est :", point_critique)


#3.4 avec sum

def lufactorization1(A):
    n = len(A)
    L = [[0.0] * n for i in range(n)]
    U = [[0.0] * n for i in range(n)]

    for i in range(n):
        L[i][i] = 1.0
        for j in range(i, n):
            sum_u = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = A[i][j] - sum_u

        for j in range(i+1, n):
            sum_l = sum(L[j][k] * U[k][i] for k in range(i))
            L[j][i] = (A[j][i] - sum_l) / U[i][i]

    return L, U

# Exemple d'utilisation :
A = [
    [2, 3, 1, 5],
    [6, 13, 5, 19],
    [2, 19, 10, 23],
    [4, 10, 11, 31]
]
#L, U = lufactorization(A)

#print("Matrice L:")
#for row in L:
#    print(row)

#print("\nMatrice U:")
#for row in U:
#    print(row)

def f1(x):
    return sin(x)
def genererFonction(f,interval,pas):
    xo,xfin=interval
    val=[]
    x=xo
    while x<=xfin:
        val.append(f(x))
        x+=pas
    return val
    

def lufactorization(A):
    n = len(A)
    L = [[0.0] * n for i in range(n)]
    U = [[0.0] * n for i in range(n)]

    for i in range(n):
        L[i][i] = 1.0
        for j in range(i, n):
            u_sum = 0.0
            for k in range(i):
                u_sum += L[i][k] * U[k][j]
            U[i][j] = A[i][j] - u_sum

        for j in range(i+1, n):
            l_sum = 0.0
            for k in range(i):
                l_sum += L[j][k] * U[k][i]
            L[j][i] = (A[j][i] - l_sum) / U[i][i]

    return L, U
A = [
    [4, -1, 0],
    [-1, 4, -1],
    [0, -1, 3]
]

#afficher(lufactorization(A)[0])
#afficher(lufactorization(A)[1])

#3.6

def ldu_factorization(A):
    L, U = lufactorization(A)  # La fonction lufactorization doit être définie comme précédemment
    n = len(A)
    D = [[0.0] * n for i in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i > j:
                D[i][j] = U[i][j]
                U[i][j] = 0
            elif i == j:
                D[i][j] = U[i][j]
                L[i][j] = 1
                U[i][j] = 1
            else:
                D[i][j] = 0
    
    # Maintenant, U est une matrice triangulaire supérieure avec des 1 sur la diagonale
    # D est une matrice diagonale
    # L est une matrice triangulaire inférieure avec des 1 sur la diagonale
    
    return L, D, U

# Exemple d'utilisation :

#L, D, U = ldu_factorization(A)

#print("Matrice L:")
#for row in L:
#    print(row)

#print("\nMatrice D:")
#for row in D:
#    print(row)

#print("\nMatrice U:")
#for row in U:
#    print(row)

#3.7

from math import sqrt

def cholesky(A):
    """Effectue la factorisation de Cholesky de la matrice A, qui doit être hermitienne et positive définie."""
    n = len(A)
    L = [[0.0] * n for i in range(n)]

    for i in range(n):
        for j in range(i+1):
            sum_k = sum(L[i][k] * L[j][k] for k in range(j))
            
            if i == j: #Éléments diagonaux
                L[i][j] = sqrt(A[i][i] - sum_k)
            else:
                L[i][j] = (1.0 / L[j][j] * (A[i][j] - sum_k))
    return L


afficher(cholesky(A))

#3.9

def jacobi(A, b, tolerance=1e-10, max_iterations=1000):
    """
    Résout le système d'équations linéaires Ax = b à l'aide de la méthode de Jacobi.
    A : matrice des coefficients
    b : vecteur constant
    tolerance : tolérance pour la convergence
    max_iterations : nombre maximum d'itérations
    Retourne la solution x.
    """
    # Initialisation
    x = [0 for _ in range(len(b))]
    D = [A[i][i] for i in range(len(A))]
    R = [[A[i][j] if i != j else 0 for j in range(len(A))] for i in range(len(A))]

    # Itération de Jacobi
    for _ in range(max_iterations):
        x_new = []
        for i in range(len(A)):
            sum_Rx = sum(R[i][j] * x[j] for j in range(len(A)))
            x_new.append((b[i] - sum_Rx) / D[i])
        
        # Vérification de la convergence
        if all(abs(x_new[i] - x[i]) < tolerance for i in range(len(A))):
            return x_new
        x = x_new
    
    raise ValueError("La méthode de Jacobi n'a pas convergé après le nombre maximum d'itérations.")

# Exemple d'utilisation :
#A = [
#    [5, -2, 3],
#    [-3, 9, 1],
#    [2, -1, -7]
#]
#b = [-1, 2, 3]
#x = jacobi(A, b)

#print("Solution x:")
#print(x)

#3.11

def gauss_seidel(A, b, x0=None, tol=1e-10, max_iter=1000):
    """
    Résout le système d'équations linéaires Ax = b à l'aide de la méthode de Gauss-Seidel.
    A : matrice des coefficients
    b : vecteur constant
    x0 : estimation initiale de la solution
    tol : tolérance pour la convergence
    max_iter : nombre maximum d'itérations
    Retourne la solution x.
    """
    n = len(A)
    x = x0 if x0 is not None else [0 for _ in range(n)]
    for _ in range(max_iter):
        x_new = x[:]
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        if all(abs(x_new[i] - x[i]) < tol for i in range(n)):
            return x_new
        x = x_new
    raise ValueError("La méthode de Gauss-Seidel n'a pas convergé après le nombre maximum d'itérations.")

# Exemple d'utilisation :
#A = [
#    [4, -1, 0],
#    [-1, 4, -1],
#    [0, -1, 3]
#]
#b = [1, 2, 0]
#x0 = [0, 0, 0]

#solution = gauss_seidel(A, b, x0)
#print(solution)

#valeurs_fonction=genererFonction(f,(0,6),0.1)


X0=[[1],[1]]
def Hessian(X):
    return [[2,0],[0,2]]
def f(X):
    return X[0][0]*X[0][0]+X[1][0]*X[1][0]

def grad(X):
    return [2*X[0][0],2*X[1][0]]
def FOX_NEWTON2(X0,epsilon,grad,Hessian,f,beta,alpha):return 0
def FOX_NEWTON(X0,epsilon,grad,Hessian,f,beta,alpha):
    rho=0.5
    X=[[X0[i][0]] for i in range(len(X0))]
    print(f"valeur de X0 :{X}")
    deltaX=multiplier3(TROUVERMATRICEINVERSE(Hessian(X)),transpose(mult(grad(X),-1)))
    print(f"deltaX :{deltaX}")
    t=mult(multiplier3([grad(X)],multiplier3(TROUVERMATRICEINVERSE(Hessian(X)),transpose(grad(X))))[0],0.5)[0]
    while t >epsilon :
        t=mult(multiplier3([grad(X)],multiplier3(TROUVERMATRICEINVERSE(Hessian(X)),transpose(grad(X))))[0],0.5)[0]
        #print(f"t : {t}")
        deltaX=multiplier3(TROUVERMATRICEINVERSE(Hessian(X)),transpose(mult(grad(X),-1))) #calcul de delta X
        print(f"deltaX :{deltaX}")
        #choix du pas 
        #g=f(add(X,multcol(deltaX,rho)))
        #h=f(X)+multiplier3([grad(X)],deltaX)[0][0]*alpha*beta
        rho=0.25
        print(f"valeur de X :{X}")
        #mise a jour de X
        #X=add(X,multcol(deltaX,rho))
        for i in range(len(deltaX)):
            for j in range(len(deltaX[0])):
              X[i][j]=X[i][j]+deltaX[i][j]*rho  # type: ignore
        
    print(f"valeur de X finale :{X}")
    return X
#FOX_NEWTON(X0,0.00001,grad,Hessian,f,0.5,0.25)



def sor(A, b, omega, initial_guess, tolerance=1e-10, max_iterations=1000):
   
    x = initial_guess
    for iteration in range(max_iterations):
        x_new = x.copy()
        for i in range(len(b)):
            sum1 = sum(A[i][j] * x_new[j] for j in range(i))
            sum2 = sum(A[i][j] * x[j] for j in range(i + 1, len(b)))
            x_new[i] = (1 - omega) * x[i] + (omega / A[i][i]) * (b[i] - sum1 - sum2)
        
        # Vérification de la convergence
        if all(abs(x_new[i] - x[i]) < tolerance for i in range(len(b))):
            return x_new
        x = x_new
    
    raise ValueError("La méthode SOR n'a pas convergé .")


A = [
    [4, 1, 2],
    [1, 3, 2],
    [2, 2, 5]
]
b = [5, 6, 8]
initial_guess = [0, 0, 0]
omega = 1.25  #Facteur de relax

#solution = sor(A, b, omega, initial_guess)
#print(solution)


def iterative_refinement(A, b, x0, tol=1e-10, max_iter=100):
  
    x = x0
    for _ in range(max_iter):
        r = [b[i] - sum(A[i][j] * x[j] for j in range(len(A))) for i in range(len(A))]
        c = [0 for _ in range(len(b))]  # Cette ligne doit être remplacée par la résolution de Ac = r
        x_new = [x[i] + c[i] for i in range(len(x))]
        
        # Vérification de la convergence
        if max(abs(x_new[i] - x[i]) for i in range(len(x))) < tol:
            return x_new
        x = x_new
    raise ValueError("Le raffinement itératif n'a pas convergé .")


A = [
    [3, 1],
    [1, 2]
]
b = [5, 5]
x0 = [0, 0]

#solution = iterative_refinement(A, b, x0)
#print(solution)



