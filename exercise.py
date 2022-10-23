"""
Justin Chen
Problem Set 3: Matrix Operation
"""
import timeit
import numpy as np

# 1)

def generate_s_matrix(n_rows):
    """Generates an NxN matrix where all edges are 1 and
    all inner values are 0 using numpy.

    Parameters
    ==========
    n_rows: int; the number of rows of the matrix

    Returns
    =======
    an n x n matrix
    """
    result = np.ones((n_rows, n_rows))
    result[1:-1, 1:-1] = 0

    return result.astype(int)

# 2)

def generate_matrix(rows, cols):
    """Generate a rows x cols matrix with random data using numpy

    Parameters
    ==========
    rows: int; the number of rows of the matrix
    cols: int; the number of colums of the matrix

    Returns
    =======
    a rows x cols matrix
    """
    return np.random.normal(size=(rows, cols))

# 3)

def matrix_multiplication_loop(x_matrix, y_matrix):
    """Multiplies x and y using a manual for-loop.

    Parameters
    ==========
    x_matrix: array; the first matrix for multiplication
    y_matrix: array; the second matrix for multiplication

    Returns
    =======
    a matrix after multiplication
    """
    result = []
    x_rows = len(x_matrix)
    y_cols = len(y_matrix[0])
    x_cols = len(x_matrix[0])
    for i in range(x_rows):
        row = []
        for j in range(y_cols):
            product = 0
            for k in range(x_cols):
                product += x_matrix[i][k] * y_matrix[k][j]
            row.append(product)

        result.append(row)
    return result

# 4)

def timed_multiplication_loop(x_matrix, y_matrix):
    """Times how long it takes to multiply x and y.
    100 matrix time: 0.558820293s
    10 matrix time: 0.0006410579999999999s

    Parameters
    ==========
    x_matrix: array; the first matrix for multiplication
    y_matrix: array; the second matrix for multiplication

    Returns
    =======
    a matrix after multiplication with loop
    the number of seconds it takes
    """
    result = matrix_multiplication_loop(x_matrix, y_matrix)
    time = timeit.timeit(lambda: matrix_multiplication_loop(x_matrix, y_matrix), number = 1)

    return result, time

# 5)

def timed_multiplication_numpy(x_matrix, y_matrix):
    """Times how long it takes to multiply x and y.
    100 matrix time: 7.02959999998587e-05s
    10 matrix time: 8.636000000006305e-06s

    Parameters
    ==========
    x_matrix: array; the first matrix for multiplication
    y_matrix: array; the second matrix for multiplication

    Returns
    =======
    a matrix after multiplication with numpy's multiplication builtins
    the number of seconds it takes
    """
    result = np.dot(x_matrix, y_matrix)
    time = timeit.timeit(lambda: np.dot(x_matrix, y_matrix), number = 1)
    return result, time
