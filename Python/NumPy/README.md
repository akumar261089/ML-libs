# <p align=center>NumPy</p>
> This Page is under construction.



## What is NumPy

NumPy specializes in numerical processing through multi-dimensional nDarrays, where the arrays allow element-by-element operations(broadcasting), linear algebra formalism.

## Benifits of NumPy array
Memory-efficient(as you can reshape data in different dimentions of array without creating multiple copies) container that provides fast numerical operations. 
Operating on the elements in a list can only be done through iterative loops, which is computationally inefficient in Python.

## Installing NumPy
apt | yum | pip
----|--------|--------------
`sudo apt-get install python-numpy`  | ` sudo yum install numpy ` |  `pip install numpy`

#### Data types for np arrays

The ndarray is similar to lists, but rather than being highly flexible by storing different types of objects in one list, only the same type of element can be stored in each column all elements must be floats, integers, or strings, it can also have elements like a list and in that case we will call it a [Structured array](#structured-array)

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
bytes | This is used for characters and strings( like a10 or S10/U10 for 10 characters long) | a10

## Structured Array

Structured arrays or record arrays are ndarrays whose datatype is a composition of simpler datatypes organized as a sequence of named fields.
scenarios like this often lend themselves to the use of Pandas DataFrames

 
``` 
>>>recarr = np.zeros((2,), dtype=('i4,f4,a10'))
>>>toadd = [(1,2.,'Hello'),(2,3.,"World")]
>>>recarr[:] = toadd
>>>recarr 

array([(1, 2., b'Hello'), (2, 3., b'World')],
      dtype=[('f0', '<i4'), ('f1', '<f4'), ('f2', 'S10')])
```


## How to create NumPy Array

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



## NumPy Array Attributes

``` 
>>>   
>>> x
array([[[8, 7, 2, 7, 6],
        [1, 4, 2, 3, 5],
        [3, 6, 0, 1, 1],
        [5, 7, 5, 2, 7]],

       [[1, 8, 2, 1, 0],
        [7, 1, 1, 4, 2],
        [6, 4, 2, 8, 4],
        [1, 8, 8, 7, 1]],

       [[2, 5, 6, 0, 3],
        [5, 4, 7, 7, 3],
        [4, 2, 7, 0, 4],
        [9, 1, 9, 2, 8]]])
```
Types| Description | examples| Output
-----|---------|---------|--------
ndarray.ndim | Dimentions of an array|  `x.ndim`|`3`
ndarray.shape | Shape of an array(can be used to reshape) |  `x.shape` |`(3, 4, 5)`
ndarry.size | Number of elements in array| `x.size`|`60`
numpy.dtype |Data type|  `x.dtype` | `dtype('int64')`
numpy.astype(int)| Change data type |` x.astype(float)` | `array([[[9., 4., 5., 8., 2.],...]`
np.around(a) | remove decimal part | `np.around(np.array([1.2, 1.5]))` | `array([1., 2.])`
numpy.flags | Show flags | `x.flags` | `C_CONTIGUOUS : True`
  |||`F_CONTIGUOUS : False`
  |||`OWNDATA : True`
  |||`WRITEABLE : True`
  |||`ALIGNED : True`
  |||`WRITEBACKIFCOPY : False`
  |||`UPDATEIFCOPY : False`

## How to move data back and forth from list to numpy

``` 
>>> a=[1,2,4]
>>> x=np.array(a)
>>> x
array([1, 2, 4])
>>> x.tolist()
[1, 2, 4]
```
 

## Numpy array Reshaping
In all reshape data will not be copied, it will be same, so if we change any value in array it will also change value in original array. So if we want a copy we should use  numpy.copy
```
>>> x = np.array([[1, 2, 3],[4, 5, 6]], dtype='i1') 
>>> x
array([[1, 2, 3],
       [4, 5, 6]], dtype=int8)
>>> x.reshape((3,2 ))
array([[1, 2],
       [3, 4],
       [5, 6]], dtype=int8)
```
We can also use ravel to flattern the array
```
>>> a = np.array([[1, 2, 3], [4, 5, 6]])
>>> a.ravel()
array([1, 2, 3, 4, 5, 6])
>>> a.T
array([[1, 4],
       [2, 5],
       [3, 6]])
>>> a.T.ravel()
array([1, 4, 2, 5, 3, 6])
>>> a.shape
(2, 3)
>>> b = a.ravel()
>>> b
array([1, 2, 3, 4, 5, 6])
>>> b = b.reshape((2, 3))
>>> b
array([[1, 2, 3],
       [4, 5, 6]])
>>> a.reshape((2, -1))
array([[1, 2, 3],
       [4, 5, 6]])
 ```
 ```
 >>> z = np.array([1, 2, 3])
>>> z
array([1, 2, 3])
>>> z[:, np.newaxis]
array([[1],
       [2],
       [3]])
>>> z[np.newaxis, :]
array([[1, 2, 3]])
```

```
>>> a = np.arange(4*3*2).reshape(4, 3, 2)
>>> a.shape
(4, 3, 2)
>>> a
array([[[ 0,  1],
        [ 2,  3],
        [ 4,  5]],

       [[ 6,  7],
        [ 8,  9],
        [10, 11]],

       [[12, 13],
        [14, 15],
        [16, 17]],

       [[18, 19],
        [20, 21],
        [22, 23]]])
>>> a[0, 2, 1]
5
>>> b = a.transpose(1, 2, 0)
>>> b.shape
(3, 2, 4)
>>> b[2, 1, 0]
5
```
```
>>> a = np.arange(4)
>>> a.resize((8,))
>>> a
array([0, 1, 2, 3, 0, 0, 0, 0])
```

## Numpy aray to matrix
we need linear algebra operations,  we can convert nd array to matrix 
>ndarray objects, matrix objects can and only will be two dimensional.

```
>>>import numpy as np
>>>arr = np.zeros((3,3))
>>>arr

array([[0., 0., 0.],
  	 [0., 0., 0.],
  	 [0., 0., 0.]])
>>>mat = np.matrix(arr)
>>>mat

matrix([[0., 0., 0.],
  	  [0., 0., 0.],
 	   [0., 0., 0.]])
```

## Numpy array indeing and slicing
Numpy indexing works similar to Python lists

``` 
>>> arr
array([[1, 2],
       [3, 4]])
>>> arr[0,1]
2
>>> arr[:,1]
array([2, 4])
>>> arr[1,:]
array([3, 4])
```

Conditional indexing

``` 
>>> arr = np.arange(5)
>>> arr
array([0, 1, 2, 3, 4])
>>> index = np.where(arr > 2)
>>> index
(array([3, 4]),)
>>> new_arr = arr[index]
>>> new_arr
array([3, 4])
```
Or 

``` 
>>> arr = np.arange(5)
>>> index = arr > 2
>>> print(index)
[False False False  True  True]
>>> new_arr = arr[index]
>>> new_arr
array([3, 4])
```
## Delete numpy index
Deleting sub array
``` 
>>> arr = np.arange(5)
>>> new_arr = np.delete(arr, index)
>>> new_arr
array([0, 1, 2])
```


## Array Concatenation and Splitting
1D
```
>>> x = np.array([1, 2, 3])
>>> y = np.array([3, 2, 1])
>>> np.concatenate([x, y])
array([1, 2, 3, 3, 2, 1])
>>> z = [99, 99, 99]
>>> np.concatenate([x, y, z])
array([ 1,  2,  3,  3,  2,  1, 99, 99, 99])
```

2D
```
>>> x = np.array([[1, 2, 3],[4, 5, 6]])
>>> x
array([[1, 2, 3],
       [4, 5, 6]])
>>> np.concatenate([x, x], axis=1)
array([[1, 2, 3, 1, 2, 3],
       [4, 5, 6, 4, 5, 6]])
>>> np.concatenate([x, x], axis=0)
array([[1, 2, 3],
       [4, 5, 6],
       [1, 2, 3],
       [4, 5, 6]])
```

##  fancy indexing

pass arrays of indices in place of single scalars.
```
>>> rand = np.random.RandomState(42)
>>> x = rand.randint(100, size=10)
>>> x
array([51, 92, 14, 71, 60, 20, 82, 86, 74, 74])
>>> ind = [3, 7, 4]
>>> x[ind]
array([71, 86, 60])
>>> ind = np.array([[3, 7],[4, 5]])
>>> x[ind]
array([[71, 86],
       [60, 20]])
```

Combining fancy indexing with the other indexing schemes 
```
>>> x = np.array([[0,  1,  2,  3],[4,  5,  6,  7],[8,  9, 10, 11]])
>>> x
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> x[2, [2, 0, 1]]
array([10,  8,  9])
>>> x[1:, [2, 0, 1]]
array([[ 6,  4,  5],
       [10,  8,  9]])

```

## Element wise  operation
Each elemet is getting affected

```
>>> a = np.array([1, 2, 3, 4])
>>> 2**a
array([ 2,  4,  8, 16])
>>> b = np.ones(4) + 1
>>> b
array([2., 2., 2., 2.])
>>> a - b
array([-1.,  0.,  1.,  2.])
>>> a * b
array([2., 4., 6., 8.])
>>> j = np.arange(5)
>>> 2**(j + 1) - j
array([ 2,  3,  6, 13, 28])
```
> Array multiplication is not matrix multiplication

Matrix multiplication:
``` 
>>> a = np.array([[0,  1,  2],[4,  5,  6],[  9, 10, 11]])
>>> a*a
array([[  0,   1,   4],
       [ 16,  25,  36],
       [ 81, 100, 121]])
>>> a.dot(a)
array([[ 22,  25,  28],
       [ 74,  89, 104],
       [139, 169, 199]])
```

```
>>> a = np.array([1, 2, 3, 4])
>>> b = np.array([4, 2, 2, 4])
>>> a == b
array([False,  True, False,  True])
>>> a > b
array([False, False,  True, False])
>>> c = np.array([1, 2, 3, 4])
>>> np.array_equal(a, b)
False
>>> np.array_equal(a, c)
True

```

```
>>> a = np.array([1, 1, 0, 0], dtype=bool)
>>> b = np.array([1, 0, 1, 0], dtype=bool)
>>> np.logical_or(a, b)
array([ True,  True,  True, False])
>>> np.logical_and(a, b)
array([ True, False, False, False])
```

```
>>> a = np.arange(5)
>>> np.sin(a)
array([ 0.        ,  0.84147098,  0.90929743,  0.14112001, -0.7568025 ])
>>> np.log(a)
__main__:1: RuntimeWarning: divide by zero encountered in log
array([      -inf, 0.        , 0.69314718, 1.09861229, 1.38629436])
>>> 
>>> 
>>> 
>>> a = np.triu(np.ones((3, 3)), 1)
>>> a
array([[0., 1., 1.],
       [0., 0., 1.],
       [0., 0., 0.]])
>>> a.T
array([[0., 0., 0.],
       [1., 0., 0.],
       [1., 1., 0.]])


```
> The transposition is a view not for matrix 


```>>> img1 = np.zeros((5, 5)) + 3
>>> img1
array([[3., 3., 3., 3., 3.],
       [3., 3., 3., 3., 3.],
       [3., 3., 3., 3., 3.],
       [3., 3., 3., 3., 3.],
       [3., 3., 3., 3., 3.]])
>>> img1[4:-4, 4:-4] = 6
>>> img1
array([[3., 3., 3., 3., 3.],
       [3., 3., 3., 3., 3.],
       [3., 3., 3., 3., 3.],
       [3., 3., 3., 3., 3.],
       [3., 3., 3., 3., 3.]])
>>> img1 = np.zeros((5, 5)) + 3
>>> img1
array([[3., 3., 3., 3., 3.],
       [3., 3., 3., 3., 3.],
       [3., 3., 3., 3., 3.],
       [3., 3., 3., 3., 3.],
       [3., 3., 3., 3., 3.]])
>>> img1[1:-1, 1:-1] = 6
>>> img1
array([[3., 3., 3., 3., 3.],
       [3., 6., 6., 6., 3.],
       [3., 6., 6., 6., 3.],
       [3., 6., 6., 6., 3.],
       [3., 3., 3., 3., 3.]])
>>> img1[2:-2, 2:-2] = 9
>>> img1
array([[3., 3., 3., 3., 3.],
       [3., 6., 6., 6., 3.],
       [3., 6., 9., 6., 3.],
       [3., 6., 6., 6., 3.],
       [3., 3., 3., 3., 3.]])
```
```
>>> index1 = img1 > 2
>>> index1
array([[ True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True]])
>>> index2 = img1 < 6
>>> index2
array([[ True,  True,  True,  True,  True],
       [ True, False, False, False,  True],
       [ True, False, False, False,  True],
       [ True, False, False, False,  True],
       [ True,  True,  True,  True,  True]])
>>> compound_index = index1 & index2
>>> compound_index
array([[ True,  True,  True,  True,  True],
       [ True, False, False, False,  True],
       [ True, False, False, False,  True],
       [ True, False, False, False,  True],
       [ True,  True,  True,  True,  True]])
>>> compound_index = (img1 > 3) & (img1 < 7)
>>> compound_index
array([[False, False, False, False, False],
       [False,  True,  True,  True, False],
       [False,  True, False,  True, False],
       [False,  True,  True,  True, False],
       [False, False, False, False, False]])
>>> img2 = np.copy(img1)
>>> img2[compound_index] = 0
>>> img2
array([[3., 3., 3., 3., 3.],
       [3., 0., 0., 0., 3.],
       [3., 0., 9., 0., 3.],
       [3., 0., 0., 0., 3.],
       [3., 3., 3., 3., 3.]])
```
```
>>> index3 = img1 == 9
>>> index3
array([[False, False, False, False, False],
       [False, False, False, False, False],
       [False, False,  True, False, False],
       [False, False, False, False, False],
       [False, False, False, False, False]])
>>> index4 = (index1 & index2) | index3
>>> index4
array([[ True,  True,  True,  True,  True],
       [ True, False, False, False,  True],
       [ True, False,  True, False,  True],
       [ True, False, False, False,  True],
       [ True,  True,  True,  True,  True]])
>>> img3 = np.copy(img1)
>>> img3[index4] = 0
>>> img3
array([[0., 0., 0., 0., 0.],
       [0., 6., 6., 6., 0.],
       [0., 6., 0., 6., 0.],
       [0., 6., 6., 6., 0.],
       [0., 0., 0., 0., 0.]])
```

## Basic Reductions or Aggregations

```
>>> x = np.array([1, 2, 3, 4])
>>> np.sum(x)
10
>>> x = np.array([[1, 1], [2, 2]])
>>> x
array([[1, 1],
       [2, 2]])
>>> x.sum(axis=0)
array([3, 3])
>>> x.sum(axis=1)
array([2, 4])

```
```
>>> x = np.random.rand(2, 2, 2)
>>> x
array([[[0.97353313, 0.23882437],
        [0.10007779, 0.90065641]],

       [[0.21706707, 0.18002739],
        [0.20039856, 0.53121455]]])
>>> x.sum(axis=2)[0, 1]
1.0007341939843606
>>> x[0, 1, :].sum()
1.0007341939843606
```

```
>>> x = np.array([1, 3, 2])
>>> x.min()
1
>>> x.max()
3
>>> x.argmin()
0
>>> x.argmax()
1

>>> np.all(a != 4)
True
>>> np.any(a >0 )
True
>>> np.any(a <0 )
False
```
```
>>> x = np.array([1, 2, 3, 1])
>>> y = np.array([[1, 2, 3], [5, 6, 1]])
>>> x.mean()
1.75
>>> np.median(x)
1.5
>>> np.median(y, axis=-1)
array([2., 5.])
>>> x.std()
0.82915619758885
```
## Broadcasting

```
>>> a = np.tile(np.arange(0, 40, 10), (3, 1)).T
>>> a
array([[ 0,  0,  0],
       [10, 10, 10],
       [20, 20, 20],
       [30, 30, 30]])
>>> 
>>> b = np.array([0, 1, 2])
>>> b
array([0, 1, 2])
>>> a + b
array([[ 0,  1,  2],
       [10, 11, 12],
       [20, 21, 22],
       [30, 31, 32]])
```

```
>>> a = np.ones((4, 5))
>>> a[0] = 2
>>> a
array([[2., 2., 2., 2., 2.],
       [1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.]])

```
```
>>> a = np.arange(0, 40, 10)
>>> a
array([ 0, 10, 20, 30])
>>> a = a[:, np.newaxis]
>>> a.shape
(4, 1)
>>> a
array([[ 0],
       [10],
       [20],
       [30]])
>>> a = np.arange(0, 40, 10)
>>> a
array([ 0, 10, 20, 30])
>>> a.shape
(4,)
>>> b = a[:, np.newaxis]
>>> b.shape
(4, 1)
>>> b
array([[ 0],
       [10],
       [20],
       [30]])
>>> a + b
array([[ 0, 10, 20, 30],
       [10, 20, 30, 40],
       [20, 30, 40, 50],
       [30, 40, 50, 60]])
```

## Sorting 

```
>>> a = np.array([[4, 3, 5], [1, 2, 1]])
>>> a
array([[4, 3, 5],
       [1, 2, 1]])
>>> np.sort(a, axis=0)
array([[1, 2, 1],
       [4, 3, 5]])
>>> np.sort(a, axis=1)
array([[3, 4, 5],
       [1, 1, 2]])
```
With Fancy indexing
```
>>> a = np.array([4, 3, 1, 2])
>>> a
array([4, 3, 1, 2])
>>> j = np.argsort(a)
>>> j
array([2, 3, 1, 0])
>>> a[j]
array([1, 2, 3, 4])
```

```
>>> a
array([4, 3, 1, 2])
>>> j_max = np.argmax(a)
>>> j_min = np.argmin(a)
>>> j_max, j_min
(0, 2)
```


## Numpy read write text and binary

```
>>> x = y = z = np.arange(0.0,5.0,1.0)
>>> x
array([0., 1., 2., 3., 4.])
>>> np.savetxt('test.out', x, delimiter=',') 
>>> arr = np.loadtxt('test.out')
>>> arr
array([0., 1., 2., 3., 4.])
>>> np.savetxt('test.out', (x,y,z))
>>> arr = np.loadtxt('test.out')
>>> arr
array([[0., 1., 2., 3., 4.],
       [0., 1., 2., 3., 4.],
       [0., 1., 2., 3., 4.]])

```

```
>>> data = np.empty((1000, 1000))
>>> np.save('test.npy', data)
>>> np.savez('test.npz', data)
>>> newdata = np.load('test.npy')
>>> newdata
array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]])
>>> newdata = np.load('test.npz')
>>> sorted(newdata.files)
['arr_0']
>>> newdata['arr_0']
array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]])
```
## Math functions
```
>>> A = np.matrix([[3, 6, -5], [1, -3, 2],
... [5, -1, 4]])
>>> A
matrix([[ 3,  6, -5],
        [ 1, -3,  2],
        [ 5, -1,  4]])
>>> B = np.matrix([[12], [-2],
... [10]])
>>> B
matrix([[12],
        [-2],
        [10]])
>>> X = A ** (-1) * B
>>> X
matrix([[1.75],
        [1.75],
        [0.75]])
```

```
>>> a = np.array([[3, 6, -5], [1, -3, 2],
... [5, -1, 4]])
>>> b = np.array([12, -2, 10])
>>> a
array([[ 3,  6, -5],
       [ 1, -3,  2],
       [ 5, -1,  4]])
>>> b
array([12, -2, 10])
>>> x = np.linalg.inv(a).dot(b)
>>> x
array([1.75, 1.75, 0.75])
```

## NumPy’s UFuncs

	
NumPy provides a convenient interface into just this kind of statically typed, compiled routine. This is known as a vectorized operation.

Operator|Equivalent ufunc|Description
----|-------|------
+ |np.add | Addition (e.g., 1 + 1 = 2) 
- |np.subtract  | Subtraction (e.g., 3 - 2 = 1)
- |np.negative |  Unary negation (e.g.,-2)
* |np.multiply |Multiplication (e.g., 2 * 3 = 6) 
/ |np.divide |Division (e.g., 3 / 2 = 1.5)
//  |np.floor_divide| Floor division (e.g., 3 // 2 = 1) 
** | np.power | Exponentiation (e.g., 2 ** 3 = 8) 
% |np.mod | Modulus/remainder (e.g., 9 % 4 = 1) 
np.abs |np.absolute|



