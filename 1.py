import numpy as np
import math

#Q1 usage of methods such as floar(),ceil(),sqrt(),isqrt(),gcd() 
print(np.floor(2.678))
print(np.ceil(2.678))
print(np.sqrt(49)) 
print(math.isqrt(8))
print(np.gcd(10,20))

#Q2 usage of attributes of array such as ndim,shape,size,methods such as sum(),mean(),sqrt(),sin() etc.
arr=np.array([1,2,3,4,5])
print(arr.ndim)
print(arr.shape)
print(arr.size)
print(np.sum(arr))
print(np.mean(arr))
print(np.sqrt(arr))
arr2=np.array([np.pi/2,np.pi/3,np.pi/4,np.pi/5])
print(np.sin(arr2))

#Q3 usage of methods such as det() and eig() 
matrix_a=np.array([[2,1],[1,2]]) 
determinant=np.linalg.det(matrix_a) 
print(determinant) 

matrix_b=np.array([[4,1],[2,3]]) 
eigenvalue,eigenvectors=np.linalg.eig(matrix_b) 
print("eif_value:",eigenvalue)
print("eif_vector:",eigenvectors)

inverse=np.linalg.inv(matrix_a)
print("inverse:",inverse)

#Q4 consider a list datatype(1D) then reshape it into 2D,3D matrix using numpy.
arr=np.array([1,2,3,4,5,6,7,8,9,10,11,12])
print(arr,arr.shape,arr.ndim)

arr_2D=arr.reshape(4,3)
print(arr_2D,arr_2D.shape,arr_2D.ndim)

arr_3D=arr.reshape(2,3,2)
print(arr_3D,arr_3D.shape,arr_3D.ndim)


#Q5 generater and ommatricesusing numpy 


#Q6 find the determinant of a matrix using Scipy 
import numpy as np
from scipy import linalg

A = np.array([[1, 2, 9],
              [5, 4, 3],
              [1, 5, 3]])

determinant = linalg.det(A)


print(f"The matrix is:\n{A}")
print(f"The determinant is: {determinant}")


#Q7 find eigen value and eigen vector of a matrix using Scipy 
from scipy.linalg import eig 
A=np.array([[1,2],
            [3,4]]) 
eig_val,eig_vector=eig(A)
print(f"eigen values are:{eig_val}")
print(f"eigen vectors are:{eig_vector}")
