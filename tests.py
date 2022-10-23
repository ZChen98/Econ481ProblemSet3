from exercise import *
from ols import *
# Tests matrix multiplication loop
test_x_matrix_10 = np.random.normal(size=(10, 10))
test_y_matrix_10 = np.random.normal(size=(10, 10))
result_loop_10, time_loop_10 = timed_multiplication_loop(
    test_x_matrix_10, test_y_matrix_10)
print('10x10 matrix time is: ' + str(time_loop_10))

test_x_matrix_100 = np.random.normal(size=(100, 100))
test_y_matrix_100 = np.random.normal(size=(100, 100))
result_loop_100, time_loop_100 = timed_multiplication_loop(
    test_x_matrix_100, test_y_matrix_100)
print('100x100 matrix time is: ' + str(time_loop_100))

# Tests matrix multiplication numpy
result_numpy_10, time_numpy_10 = timed_multiplication_numpy(
    test_x_matrix_10, test_y_matrix_10)
result_numpy_100, time_numpy_100 = timed_multiplication_numpy(
    test_x_matrix_100, test_y_matrix_100)
print('10x10 matrix time is: ' + str(time_numpy_10))
print('100x100 matrix time is: ' + str(time_numpy_100))

print(extract_estimator('campus.csv'))
print(extract_variable_means('campus.csv'))
