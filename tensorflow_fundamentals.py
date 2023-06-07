# Introduction to Tensors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# print(tf.__version__)


# Create tensors with tf.constant()
scalar = tf.constant(7)
# print(scalar)


# Check the number of dimensions of a tensor (ndim stands for number of dimensions)
check_dim_vector = scalar.ndim
# print(var)


# Create a vector
vector = tf.constant([10, 10])
# print(vector)


# Check the dimension of our vector
check_dim_vector = vector.ndim
# print(check_dim_vector)


# Create a matrix (has more than 1 dimension)
matrix = tf.constant([[10, 7],
                     [7, 10]])
# print(matrix)


# Check dimension of matrix above

check_matrix_ndim = matrix.ndim
# print(check_matrix_ndim)


# Create another matrix
another_matrix = tf.constant([[3., .10],
                               [7., .5],
                               [5., .3]], dtype=tf.float16)
# print(another_matrix)


# what's the number dimensions of another matrix?
# print(another_matrix.ndim)


# let's create a tensor
tensor = tf.constant([[[1, 2, 3],
                       [4, 5, 6]],
                       [[7, 8, .9],
                        [10, 11, 12]],
                        [[13, 14, 15],
                        [16, 17, 18]]])
# print(tensor)
# print(tensor.ndim) #checking the number dimensions of another matrix above

# summary what already created
# Scalar: a single number
# Vector: a number with direction(e.g. wind speed and direction)
# Matrix: a 2-dimensional array of numbers
# Tensor: an n-dimensional array of numbers (when n can be any number, a 0-dimensional tensor is a scalar,
#         a 1-dimensional tensor is a vector)


# Create the same tensor with tf.variable() as above

changeable_tensor = tf.Variable([10, 7])
unchangeable_tensor = tf.constant([10, 7])
# print(changeable_tensor, unchangeable_tensor)


# let's try change one of the elements in our changable tensor
# changeable_tensor[0] = 7
# print(changeable_tensor)


# How about to try .assign()
Changable = changeable_tensor[0].assign(7)
# print(Changable)


# Try unchangable tensor
# unchangeable = unchangeable_tensor[0].assign(7)
# print(unchangeable)


# Creating random tensors (random tensors are tensors of some abitrary size which contain random numbers)
random_1 = tf.random.Generator.from_seed(42)  # set seed for reproducibility
random_1 = random_1.normal(shape=(3, 2))
random_2 = tf.random.Generator.from_seed(42)
random_2 = random_2.normal((3, 2))
# print(random_1)

# Checking equal or not
check = random_1, random_2, random_1 == random_2
# print(check)

# Shuffle a tensor (valuable for when you want to shuffle your data so the inherent order doesn't effect learning)
not_shuffled = tf.constant([[10, 7],
                             [3, 4],
                             [2, 5]])
# print(not_shuffled.ndim)
# print(not_shuffled)


# shuffle non-shuffled tensor
non_shuffled = tf.random.shuffle(not_shuffled)
# print(non_shuffled)


# Exercise for own: Read through TensorFlow documentation on random seed generation:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed and practice writing random tensors and shuffle them.
# rule 4: "If both the global and the operation seed are set: Both seeds are used in
#          conjunction to determine the random sequence."


tf.random.set_seed(42)  #global level random seed
operation_level_random = tf.random.shuffle(not_shuffled, seed=42)  # operation level random seed
# print(operation_level_random)


# Otherways to make tensors
tensor_one = tf.ones([10, 7])
# print(tensor_one)

# Create a tensor of all zeroes
tensor_zero = tf.zeros(shape=(10, 7))
# print(tensor_zero)


# Turn NumPy arrays into tensors
# "The main difference between NumPy arrays and TensorFlow tensors is that tensors can be run on a GPU
# (much faster for numerical computing)."

# Turn NumPy arrays into tensors
import numpy as np
numpy_A = np.arange(1, 25, dtype=np.int32) #create a NumPy array between 1 and 25
# print(numpy_A)

# X = tf.constant(some_matrix) #capital for matrix or tensor
# y = tf.constant(vector) #non-capital for vector

# making shape of tf.constant
A = tf.constant(numpy_A, shape=(2, 3, 4))

# no shape of tf.constant
B = tf.constant(numpy_A)
# print(A, B)


# Getting information from tensors

# When dealing with tensors you probably want to be aware of the following attributes:

# Shape, the length (number of elements) of each of the dimensions of a tensor (code = tensor.shape).

# Rank, the number of tensor dimensions. A scalar has rank 0, a vector has rank 1, a matrix is rank 2,
# a tensor has rank n. (code = tensor.ndim).

# Axis or Dimension, a particular dimension of a tensor. (code = tensor[0], tensor[:, 1]).

# Size, the total number of items in the tensor (code = tf.size(tensor))


# Create a rank 4 tensor (4 dimensions)
rank_4_tensor = tf. zeros(shape=[2, 3, 4, 5])
# print(rank_4_tensor)

# Create a rank 4 tensor from index
# print(rank_4_tensor[0])

# checking shape, rank, and size of rank_4_tensor
rank_4 = rank_4_tensor.shape, rank_4_tensor.ndim, tf.size(rank_4_tensor)
# print(rank_4)


# Get various attributes of our tensors
# print("Datatype of every element:", rank_4_tensor.dtype)
# print("Number of dimensions (rank):", rank_4_tensor.ndim)
# print("Shape of tensor:", rank_4_tensor.shape)
# print("Elements along the 0 axis:", rank_4_tensor.shape[0])
# print("Elements along the last axis:", rank_4_tensor.shape[-1])
# print("total number of elements in our tensor:", tf.size(rank_4_tensor))
# print("total number of elements in our tensor:", tf.size(rank_4_tensor).numpy())


""""Indexing tensor"""
# Tensors can be indexed just like Python lists.


some_list = [1, 2, 3, 4]
# print(some_list[:2])


# Get the first 2 elements of each dimension
rank_4_tensor_ = rank_4_tensor[:2, :2, :2, :2]
# print(rank_4_tensor_)


# Get the first element from each dimension from each index except for the final one
rank_4_tensor = rank_4_tensor[:1, :1, :1, :]
# print(rank_4_tensor)


# Create a rank 2 tensor (2 dimensions)
rank_2_tensor = tf.constant([[10, 7],
                            [3, 4]])
print(rank_2_tensor, rank_2_tensor.ndim)

# print(some_list, some_list[-1])


# Get the last item of each of row of our rank 2 tensor
# print(rank_2_tensor[:, -1])


# Add in extra dimension to our rank 2 tensor
rank_3_tensor = rank_2_tensor[..., tf.newaxis]
# print(rank_3_tensor)


# Alternative to tf.newaxis
expand_dimension = tf.expand_dims(rank_2_tensor, axis=0) # "-1" means expand the final axis
# print(expand_dimension)


"""Manipulating Tensors (tensor operations)"""
# Basic operations ( +, -, *, /)

# You can add values to a tensor using the addition operator
tensor = tf.constant([[10, 7],
                      [3, 4]])
# Original tensor
# print(tensor)

# using '+'
# print("using '+' tensor:", tensor + 10)

# using '-' substraction
# print("using '-' tensor:", tensor - 10)

# using '*' multiple
# print("using '*' tensor:", tensor * 10)

# using '/' dividing
# print("using '/' tensor:", tensor / 10)


# We can use the tensorflow built-in function too
tf.multiply(tensor, 10)


"""Matrix Multiplication"""
# In machine learning. matrix multiplication is one of the most common tensor operations
# there are two rules our tensors (or matrices) need to fulfill if we're going
# to matrix multiply them :

# 1. The inner dimensions must match
# 2. The resulting matrix has the shape of the inner dimensions

# Matrix multipication in tensorflow
multipication = tf.matmul(tensor, tensor)
# print("hasil dari multipiclation: ", multipication)
# print("hasil dari tensor: ", tensor)


# Multiplying tensor
multiply = tensor * tensor
# print(multiply)


# Matrix multipication with Python operator "@"
matrix_multiply = tensor @ tensor
# print("hasil dari perkalian :", matrix_multiply)


# Create a tensor (3, 2)
X = tf.constant([[1, 2],
                [3, 4],
                [5, 6]])
# Create another tensor (3, 2)
Y = tf.constant([[7, 8],
                 [9, 10],
                 [11, 12]])
# print("Hasil tensor pertama: ",X ,"Hasil tensor kedua: ", Y)


# Try to matrix multiply tensors of same shape
# try_ = tf.matmul(X, Y)
# print(try_)


# Let's change the shape of Y
changing_shape = tf.reshape(Y, shape=(2, 3))
print(changing_shape)
