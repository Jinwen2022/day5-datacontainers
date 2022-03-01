#a-e
import numpy as np
from scipy import linalg
A = np.array([(1,2,3),(4,5,6),(7,8,9)])
b = np.array([1,2,3])
x= linalg.solve(A, b)
print('x:',x)
print('A*x=',np.dot(A,x),'b=',b)

#e
b =  np.random.random((3,3))
la,v = linalg.eig(A)
l1, l2,l3 = la
print('eigenvalues:',l1,l2,l3)
print('first eigenvector:',v[:, 0])
print('second eigenvector:',v[:, 1])
print('Third eigenvector:',v[:, 2])

#f
determinant = linalg.det(A)
A_1 =  linalg.inv (A)
print('determinant:',determinant)
print('Inverse:',A_1)

#h. Calculate the norm of A with different orders
print('order:2',linalg.norm(A))
print('order:1',linalg.norm(A,1))
print('order:-2',linalg.norm(A,-2))
print('order:-1',linalg.norm(A,-1))



