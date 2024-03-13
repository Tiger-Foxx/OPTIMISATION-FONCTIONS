# 2.2

import copy
import random

from sympy import N

from main import afficher, multiplier2


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

from math import comb

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
        
def minus(A,B):
    return [a - b for a, b in zip(A, B)]
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


TROUVERMATRICEINVERSE(matrice_exemple)
multiplier2(TROUVERMATRICEINVERSE(matrice_exemple),matrice_exemple)




