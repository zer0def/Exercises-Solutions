#!/usr/bin/env python
#
# Display Device Information
#
# Script to print out some information about the OpenCL devices
# and platforms available on your system
#
# History: C++ version written by Tom Deakin, 2012
#          Ported to Python by Tom Deakin, July 2013
#

# Import the Python OpenCL API
import pyopencl as cl

# Create a list of all the platform IDs
platforms = cl.get_platforms()

print("Number of OpenCL platforms: {0}".format(len(platforms)))

PLATFORM_FMT = """-------------------------
Platform: {name}
Vendor: {vendor}
Version: {version}
Number of devices: {device_count}"""

DEVICE_FMT = """  -------------------------
    Name: {name}
    Version: {version}
    Max. Compute Units: {max_computes}
    Local Memory Size: {local_mem_size} KB
    Global Memory Size: {global_mem_size} MB
    Max Alloc Size: {max_alloc_size} MB
    Max Work-group Total Size: {max_work_group_size}
    Max Work-group Dimensions: {max_work_group_dims}"""

# Investigate each platform
for p in platforms:
    # Discover all devices
    devices = p.get_devices()

    # Print out some information about the platforms
    print(PLATFORM_FMT.format(
        name=p.name,
        vendor=p.vendor,
        version=p.version,
        device_count=len(devices)
    ))

    # Investigate each device
    for d in devices:
        print(DEVICE_FMT.format(
            name=d.name,
            version=d.opencl_c_version,
            max_computes=d.max_compute_units,
            local_mem_size=d.local_mem_size/2**10,
            global_mem_size=d.global_mem_size/2**20,
            max_alloc_size=d.max_mem_alloc_size/2**20,
            max_work_group_size=d.max_work_group_size,
            max_work_group_dims=d.max_work_item_sizes
        ))
