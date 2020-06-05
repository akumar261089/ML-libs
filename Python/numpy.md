# <p align=center>NumPy</p>


### Introduction 
> In this section is under construction.

##### What is NumPy

NumPy specializes in numerical processing through multi-dimensional ndarrays, where the arrays allow element-by-element operations(broadcasting), linear algebra formalism can be used without modifying the NumPy arrays before- hand. the arrays can be modified in size dynamically

##### Benifits of numpy array
Memory-efficient container that provides fast numerical operations.

##### How to create numpy array


##### Data types
#####Structured Array

##### How to move data back and forth from list to numpy



##### Numpy array Reshaping

##### Numpy aray to matrix

##### Numpy array indeing and slicing

##### conditional and fancy indexing
numpy.where()

##### Element wise  operation

Basic Reductions

Broadcasting

Sorting 
More elaborate arrays

maskedarray


##### delete numpy index

##### boolean statement with numpy arrays


##### Numpy read write text and binary

##### Math functions
	Why we need numpy array as we already have Python list and dict - operating on the elements in a list can only be done through iterative loops, which is computationally inefficient in Python.
	example 
	```
	siome example code ie

	import numpy as np
	# Creating a 3D numpy array 
	arr = np.zeros((3,3)) 
	```

	The ndarray is similar to lists, but rather than being highly flexible by storing different types of objects in one list, only the same type of element can be stored in each column all elements must be floats, integers, or strings, it can also have elements like a list and in that case we will call it a Structured array
	``` 
	>>> recarr = np.zeros((2,), dtype=('i4,f4,a10'))
	>>> toadd = [(1,2.,'Hello'),(2,3.,"World")]
	>>> recarr[:] = toadd
	>>> recarr
	array([(1, 2., b'Hello'), (2, 3., b'World')],
      dtype=[('f0', '<i4'), ('f1', '<f4'), ('f2', 'S10')])
	```
	> where i4 corresponds to a 32-bit integer, f4 corresponds to a 32-bit float, and a10 corresponds to a string 10 characters long.

	we need linear algebra operations,  we can convert nd array to matrix 
	ndarray objects, matrix objects can and only will be two dimensional.

	```
    >>> import numpy as np
	# Creating a 2D numpy array 
	>>> arr = np.zeros((3,3))
	>>> arr
	array([[0., 0., 0.],
      	 [0., 0., 0.],
      	 [0., 0., 0.]])
    >>> mat = np.matrix(arr)
    >>> mat
	matrix([[0., 0., 0.],
      	  [0., 0., 0.],
     	   [0., 0., 0.]])
	```

	In all reshape data will not be copied, it will be same, so if we change any value in array it will also change value in original array. So if we want a copy we should use  numpy.copy
	```
>>> new = np.zeros((5,5))
>>> new
	array([[0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.]])
>>> new1d = new.ravel()
>>> new1d
array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0.])
>>> new1d[1]=1
>>> new1d
array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0.])
>>> new
array([[0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.]])