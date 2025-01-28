---
layout: post
title: RcppCGAL is back on CRAN
description: >
  RcppCGAL updated and back on CRAN
  sitemap: false
---
  
After a bit of a hiatus which saw the package be archived from CRAN due to a bug and their varying machine requirements, the package is back on CRAN.

## What's new
The header files are now set to a use a fixed version, per CRAN request, and the function `cgal_install()` has been deprecated. The reason is that the header files come pre-bundled with the package. One can still use the `CGAL_DIR` environmental variable to use one's own version of CGAL, if so desired. You can also use the function `set_cgal` if you're not comfortable with setting system environment variables.

There was also a bug caused by my over zealous cleaning function destroying a CGAL exit template that was needed by a reverse dependency. Thank you to Tyler Morgan-Wall for catching it! He's been added as a contributor.

## Where can you find the package?
As always, on the [github](https://www.github.com/ericdunipace/RcppCGAL) and from time to time on [CRAN](https://CRAN.R-project.org/package=RcppCGAL)

## What's next
Hopefully nothing till the next bug!


                                                             