#!/usr/bin/env python
#
# This program will numerically compute the integral of
#
#                4/(1+x*x)
#
# from 0 to 1.  The value of this integral is pi -- which
# is great since it gives us an easy way to check the answer.
#
# This the original sequential program.
#
# History: Written in C by Tim Mattson, 11/99
#          Ported to Python by Tom Deakin, July 2013
#

from time import time

num_steps = 100000000

print("Note: Wanted to do {0} steps, but this is very slow in Python.".format(num_steps))

num_steps = 1000000

print("Doing {0} steps instead.".format(num_steps))

start_time = time()

print("pi with {steps} steps is {result} in {run_time} seconds".format(
    steps=num_steps,
    result=sum([
        4.0/(1.0+((i-0.5)/float(num_steps))**2) for i in range(1, num_steps)
    ])/float(num_steps),
    run_time=time()-start_time
))
