#!/usr/bin/env python
#
# Device Info
#
# Function to output key parameters about the input OpenCL device
#
# History: C version written by Tim Mattson, June 2010
#          Ported to Python by Tom Deakin, July 2013
#

import pyopencl as cl
import sys

DEVICE_FMT = """Device is {name}
{type} from {vendor} with a max of {max_computes} compute units"""

def output_device_info(device_id):
    device_type = "unknown unit type"
    if device_id.type == cl.device_type.GPU:
        device_type = "GPU"
    elif device_id.type == cl.device_type.CPU:
        device_type = "CPU"

    print(DEVICE_FMT.format(
        name=device_id.name,
        type=device_type,
        vendor=device_id.vendor,
        max_computes=device_id.max_compute_units
    ))
