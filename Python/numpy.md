# <p align=center>NumPy</p>
> This Page is under construction.

## Introduction 


In this document I will cover basic NumPy functionalities which you can practice.

#### What is NumPy

NumPy specializes in numerical processing through multi-dimensional nDarrays, where the arrays allow element-by-element operations(broadcasting), linear algebra formalism.

#### Benifits of NumPy array
Memory-efficient(as you can reshape data in different dimentions of array without creating multiple copies) container that provides fast numerical operations. 
Operating on the elements in a list can only be done through iterative loops, which is computationally inefficient in Python.

#### Installing NumPy
apt | yum | pip
----|--------|--------------
`sudo apt-get install python-numpy`  | ` sudo yum install numpy ` |  `pip install numpy`

#### Data types for np arrays

The ndarray is similar to lists, but rather than being highly flexible by storing different types of objects in one list, only the same type of element can be stored in each column all elements must be floats, integers, or strings, it can also have elements like a list and in that case we will call it a Structured array

NumPy arrays contain values of a single type, so it is important to have detailed knowledge of those types and their limitations.

`np.zeros(10, dtype=np.int16)`
`np.zeros(10, dtype='i2')`

Data type| Description | short hand
--------|--------------|----------
bool_	| Boolean (True or False) stored as a byte | 
int_	| Default integer type (same as C long; normally either int64 or int32) | 
intc	| Identical to C int (normally int32 or int64) |
intp	| Integer used for indexing (same as C ssize_t; normally either int32 or int64)  |
int8	| Byte (–128 to 127) | i1
int16	| Integer (–32768 to 32767) | i2
int32	| Integer (–2147483648 to 2147483647) | i4
int64	| Integer (–9223372036854775808 to 9223372036854775807) |i8
uint8	| Unsigned integer (0 to 255) | u1
uint16	| Unsigned integer (0 to 65535) | u2
uint32	| Unsigned integer (0 to 4294967295) |u4
uint64	| Unsigned integer (0 to 18446744073709551615) | u8
float_	| Shorthand for float64 | f8
float16	|Half-precision float: sign bit, 5 bits exponent, 10 bits mantissa | f2
float32	| Single-precision float: sign bit, 8 bits exponent, 23 bits mantissa | f4
float64	| Double-precision float: sign bit, 11 bits exponent, 52 bits mantissa |f8
complex_ | 	Shorthand forcomplex128 | c16
complex64	| Complex number, represented by two 32-bit floats | c8
complex128	| Complex number, represented by two 64-bit floats | c16

#### Structured Array

#### How to create NumPy Array

Import NumPy in your python code: ` import numpy as np `



1D | Empty | Zeros | Sequential numbers | Specific steps between range | List
---|------|------|--------------------|---------------------------------|-------
Examples   |`np.empty(2)` | `np.zeros(3)` | `np.arange(3)` | `np.linspace(0, 1, 5)` | `np.array([3,2,1])`
Output | `array([1., 1.])`|`array([0., 0., 0.])` | `array([0, 1, 2])` | `array([0.  , 0.25, 0.5 , 0.75, 1.  ])` | `array([3, 2, 1])`
Examples |      |         | `np.arange(2,5)` | `np.logspace(0, 1, 5, base=10.0)`|
Output |     |       |`array([2, 3, 4])` | `array([ 1.        ,  1.77827941,  3.16227766,  5.62341325, 10.        ])`|



2D | Random | normal(with range) | random int | identity matrix
----------|------|--------------------|---------------------------------|-------
Examples   | `np.random.random((3, 3))` | `np.random.normal(0, 1, (2, 2))` | `np.random.randint(0, 10, (2, 2))` | `np.eye(2)`
Output | `array([[0.1978, 0.6596], [0.7029, 0.5751]])` | `array([[-0.0952,  0.0182], [-0.4957,  0.0688]])` | `array([[1, 9],  [6, 4]])` | `array([[1., 0.],[0., 1.]])`

You can specify the bit depth when creating arrays by setting the data type parameter (dtype) to int, numpy.float16, numpy.float32, or numpy.float64. 
`np.zeros(3, dtype=int)` and thios will give you `array([0, 0, 0])`
>All  methods can be used to create 1D, 2D or nD arrays by giving dimentions in (n,m) example - `np.zeros((3,3))`



##### NumPy Array Attributes
##### How to move data back and forth from list to numpy



##### Numpy array Reshaping

##### Numpy aray to matrix

##### Numpy array indeing 
##### Accessing and slicing
##### Array Concatenation and Splitting
##### conditional and fancy indexing
numpy.where()

##### Element wise  operation

##### Basic Reductions

##### Broadcasting

##### Sorting 
##### More elaborate arrays

##### maskedarray
##### Aggregations

##### delete numpy index

##### boolean statement with numpy arrays


##### Numpy read write text and binary

##### Math functions
##### NumPy’s UFuncs

	
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