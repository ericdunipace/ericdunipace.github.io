---
layout: post
title: RcppCGAL updated
description: >
  RcppCGAL has undergone some updates!
sitemap: false
---

I've done some fairly extensive changes to the [RcppCGAL package](https://github.com/ericdunipace/RcppCGAL), but since this is probably the first time I've described it, I should back up a little bit and maybe describe what it does and answer any other anticipated questions you may have.

## What does CGAL stand for?
CGAL stands for the Computational Geometry Algorithms Library. You can read more about it [here](https://www.cgal.org).

## Why have you done this?
Basically, I did this because I needed the [Hilbert sorting](https://doc.cgal.org/latest/Spatial_sorting/index.html#sechilbert_sorting) function from CGAL for the [first paper](https://arxiv.org/abs/2012.09999) of my [thesis](https://dash.harvard.edu/handle/1/37368342). It seemed like there could be a need for others to use these header files, so I made a package.

## What software is it for?
Anything running in R in need of some C/C++ header files.

## What changes have you made?
The package will try to download the latest header files and will then change the C code to make sure that any messages are printed to the R console and that any errors and stop signals are handled appropriately so that R doesn't crash.

In this specific version, I've added a function `cgal_install` to download the header files and change the outputs. This will hopefully make it easier to update the header files but also will allow users to install the header files when the automatic download on install doesn't work. 

## How do I use it?
Really, this is a bare bones package simply meant to make the process of including the latest CGAL header files into an R package easier. To use it you just need to install the package in the usual manner from R: `install.packages("RcppCGAL")`. You can also use the `remotes` package to download directly from GitHub. There's a vignette in the package that also describes how to install the header files.

To see what kinds of things are possible with CGAL, I recommend checking out Stéphane Laurent's packages. He details some of them [here](https://laustep.github.io/stlahblog/posts/SurfaceReconstruction.html).

From my own uses, I have the following example in the README file of the GitHub on how to do Hilbert sorting
```c++
// [[Rcpp::depends(RcppCGAL)]]
// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(cpp14)]]  

#include <RcppEigen.h>
#include <CGAL/basic.h>
#include <CGAL/Cartesian_d.h>
#include <CGAL/spatial_sort.h>
#include <CGAL/Spatial_sort_traits_adapter_d.h>
#include <CGAL/boost/iterator/counting_iterator.hpp>
#include <CGAL/hilbert_sort.h>
#include <CGAL/Spatial_sort_traits_adapter_d.h>

typedef CGAL::Cartesian_d<double>           Kernel;
typedef Kernel::Point_d                     Point_d;

typedef CGAL::Spatial_sort_traits_adapter_d<Kernel, Point_d*>   Search_traits_d;

void hilbert_sort_cgal_fun(const double * A, int D, int N,  int * idx)
{
  
  std::vector<Point_d> v;
  double * temp = new double[D];
  
  for (int n = 0; n < N; n++ ) {
    for (int d = 0; d < D; d ++) {
      temp[d] = A[D * n + d];
    }
    v.push_back(Point_d(D, temp, temp+D));
  }
  
  std::vector<std::ptrdiff_t> temp_index;
  temp_index.reserve(v.size());
  
  std::copy(
    boost::counting_iterator<std::ptrdiff_t>(0),
    boost::counting_iterator<std::ptrdiff_t>(v.size()),
    std::back_inserter(temp_index) );
  
  CGAL::hilbert_sort (temp_index.begin(), temp_index.end(), Search_traits_d( &(v[0]) ) ) ;
  
  for (int n = 0; n < N; n++) {
    idx[n] = temp_index[n];
  }
  
  delete [] temp;
  temp=NULL;
}

// [[Rcpp::export]]
Rcpp::IntegerVector hilbertSort(const Eigen::MatrixXd & A)
{
  int K = A.rows();
  int N = A.cols();
  std::vector<int> idx(N);
  
  hilbert_sort_cgal_fun(A.data(), K, N, &idx[0] );
  return(Rcpp::wrap(idx));
}
```

Saving this code as `hilbertSort.cpp` and sourcing with Rcpp `Rcpp::sourceCpp("hilbertSort.cpp")`
makes the function `hilbertSort()`. Be aware that this example 
function example assumes that the observations are stored by
column rather than by row, that is as the transpose of the 
usual `R` `matrix` or `data.frame`.

## What next?
Hopefully, there won't be too much more to this package now that it should autoupdate to the latest version of CGAL.
