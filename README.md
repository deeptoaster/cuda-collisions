cuda-collisions
===============

About
-----

This project provides an implementation of broad-phase collision detection via 
spatial subdivision. It contains a proof-of-concept narrow-phase collision 
testing algorithm for balls but can be extended to use any collision testing 
algorithm.


Benchmarking
------------

Run `make` to build. The syntax is

    ./collisions NUMOBJECTS MAXSPEED MAXDIM [NUMBLOCKS [THREADSPERBLOCK]


Unit Testing
------------

Run `make test` to build. The collisions_test program runs unit tests on each 
kernel component of the algorithm.


Documentation
-------------

Run `make docs` to generate documentation.
