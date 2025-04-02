import numpy as np

class NumpyML:
    def __init__(self):
        pass

    def matrix_Add(self, a, b):  
        return np.add(a, b)
    
    def matrix_Subtract(self, a, b):  
        return np.subtract(a, b)

    def matrix_Product(self, a, b):  
        return np.dot(a, b)
    
    def matrix_Transpose(self,a):
        return np.transpose(a)

    def matrix_Determine(self,a):
        return np.linalg.det(a)

    def matrix_Inverse(self,a):
        det = np.linalg.det(a)
        if det == 0:
            raise ValueError("Matrix is not invertible (determinant is zero).")
        return np.linalg.inv(a)
    def matrix_Eigen(self, A):
        eigenvalues, eigenvectors = np.linalg.eig(A)
        return eigenvalues, eigenvectors
    def matrix_SVD(self, A):
        U, S, Vt = np.linalg.svd(A)
        return U, S, Vt
    def gradient_descent(self, X, y, theta, learning_rate=0.01, iterations=1000):
        m = len(y)  
        for _ in range(iterations):
            gradient = (1/m) * np.dot(X.T, (np.dot(X, theta) - y))  
            theta -= learning_rate * gradient  
        return theta
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def logistic_regression(self, X, y, theta, learning_rate=0.01, iterations=1000):
        m = len(y)
        for _ in range(iterations):
            z = np.dot(X, theta)
            predictions = self.sigmoid(z)
            gradient = (1/m) * np.dot(X.T, (predictions - y))
            theta -= learning_rate * gradient
        return theta



ml = NumpyML()

r = np.array([1, 2])
a = np.array([2, 3])
x = np.array([[1, 2], [5, 6]])
y = np.array([[3, 4], [7, 8]])

resultA = ml.matrix_Add(r, a)
resultS = ml.matrix_Subtract(r, a)
resultP = ml.matrix_Product(x, y)
resultT = ml.matrix_Transpose(x)
resultD = ml.matrix_Determine(y)
resultI = ml.matrix_Inverse(x)


print("Addition:", resultA)
print("Subtraction:", resultS)
print("Multiplication:\n", resultP)
print("Transpose:\n",resultT)
print("Determinent:\n",resultD)
print("Inverse:",resultI)
