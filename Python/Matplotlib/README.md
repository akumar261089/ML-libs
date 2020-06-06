# <p align="center"> Matplotlib </p> 

> This page is under construction

## What is Matplotlib

Matplotlib is a multiplatform data visualization library built on NumPy arrays 
> Matplotlib is probably the most used Python package for 2D-graphics. It provides both a quick way to visualize data from Python and publication-quality figures in many formats.

## Features of Matplotlib

* Easy to get started
* Support for LATEX formatted labels and texts
* Great control of every element in a figure, including figure size and DPI.
* High-quality output in many formats, including PNG, PDF, SVG, EPS, and PGF.
* GUI for interactively exploring figures and support for headless generation of figure files (useful for batch jobs).

## Installing Matplotlib

apt | yum | pip
----|--------|--------------
`apt-get install python3-matplotlib`  | `sudo yum install python3-matplotlib` |  `pip3 install Matplotlib`

## Package import 

```
import matplotlib as mpl 
import matplotlib.pyplot as plt
```


## Matplotlib figure structure 

![Matplotlib figure structure ](images/figure-components.png)

Component | Description
----------|------------
Figure | One Figure is complete plot area which includes everything including Subplots.
Subplot | Subplot is like sub part of figure which contains all data to be displayed on same axes.
 Axis | These are the number-line-like objects. They take care of setting the graph limits and generating the ticks.
 Axes| The Axes contains two (or three in the case of 3D) Axis objects (be aware of the difference between Axes and Axis) which take care of the data limits.
 Spine|Spine are four lines that denote the boundaries of the data area.
 Grid|Grids are lines inside the data area that aid the reading of values.
 Title|A name of figure that describes the figure
 Axis Labels|Description of each axis, unit should be given if applicable.
 Ticks|Ticks are marks of division on a plot axis.Types -  Major or minor
 Tick labels|Both major and minor ticks can be labelled.
 Legend|Labels for each data series
 Patches|Different shapes can be added using `matplotlib.patches` like rectangle, circle etc.

## pyplot

pyplot provides a procedural interface to the matplotlib object-oriented plotting library. It is modelled closely after MatlabTM. Therefore, the majority of plotting commands in pyplot have MatlabTM analogy with similar arguments. 

Type of plot|	Function & Description
------------|-------------------------
Bar	|Make a bar plot.
Barh|	Make a horizontal bar plot.
Boxplot	|Make a box and whisker plot.
Hist	|Plot a histogram.
hist2d	|Make a 2D histogram plot.
Pie	|Plot a pie chart.
Plot|	Plot lines and/or markers to the Axes.
Polar|	Make a polar plot..
Scatter	|Make a scatter plot of x vs y.
Stackplot|	Draws a stacked area plot.	
Stem|	Create a stem plot.
Step	|Make a step plot.
Quiver	|Plot a 2-D field of arrows.

### Figure Functions

Function | Description
---------|-----------
Figtext|Add text to figure.
Figure| Creates a new figure.
Show| Display a figure.
Savefig| Save the current figure.
Close| Close a figure window.

### Axis Functions

Function|	Description
--------|------------------
Axes	|Add axes to the figure.
Text|	Add text to the axes.
Title|	Set a title of the current axes.
Xlabel|	Set the x axis label of the current axis.
Xlim	|Get or set the x limits of the current axes.
Xscale	|.
Xticks	|Get or set the x-limits of the current tick locations and labels.
Ylabel	|Set the y axis label of the current axis.
Ylim	|Get or set the y-limits of the current axes.
Yscale	|Set the scaling of the y-axis.
Yticks	|Get or set the y-limits of the current tick locations and labels.

### Image Functions

Function|	Description
--------|------------------
Imread| Read an image from a file into an array.
Imsave | Save an array as in image file.
Imshow | Display an image on the axes.

## Common Colors in matplotlib

Symbol | Colour
-------|------
b| Blue
g| Green
r| Red
c| Cyan
m| Magenta
y| Yellow
k| Black
w| White

More colours name - https://matplotlib.org/examples/color/named_colors.html

#### RGB or RGBA  or Hex
It will be float values from (0-1) for each Red, Green, Blues ,degree of transparency(optional)  or #ffffff




##### Simple plot or ,line space 

##### Instantiating defaults


##### Changing colors and line widths


##### Setting limits
##### Setting ticks

##### Setting tick labels
##### Moving spines

##### Adding a legend
	
##### Annotate some points

##### Devil is in the details

### Figures, Subplots, Axes and Ticks

##### Figures,
##### Subplots,
##### Axes and
##### Ticks

### Other plots

##### Regular
##### Scatter
##### Bar
##### Contour
##### Image show
##### PIE CHARTS
##### Quiver Plots
##### Grids
##### Multi Plots
##### Polar Axis
##### 3D Plots
##### Text

