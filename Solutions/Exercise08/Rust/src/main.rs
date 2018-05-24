extern crate ocl;

extern crate float_cmp;

use std::io::Read;
use float_cmp::ApproxEqRatio;

const AVAL: ocl::prm::cl_float = 3.0;
const BVAL: ocl::prm::cl_float = 5.0;
//const TOL: f32 = 0.001;
//const DIM: usize = 2;
const COUNT: usize = 2;

fn post_rustbook() -> Result<(), ocl::error::Error> {
    let order = match ocl::Context::builder()
        .build()?
        .device_info(0, ocl::enums::DeviceInfo::MaxWorkGroupSize)?
    {
        ocl::enums::DeviceInfoResult::MaxWorkGroupSize(value) => value,
        _ => panic!(""),
    } as usize;

    // Work-group computes a block of C. This size is also set
    // in a #define inside the kernel function. Note this blocksize
    // must evenly divide the matrix order
    let blocksize = 16;

    for &(title, path, global_size, local_size, local_mem) in vec![
        (
            "OpenCL, matrix mult, C(i,j) per work item",
            "../C_elem.cl",
            ocl::SpatialDims::from((order, order)),
            ocl::SpatialDims::Unspecified,
            &vec![0; 0],
        ),
        (
            "OpenCL, matrix mult, C row per work item",
            "../C_row.cl",
            ocl::SpatialDims::from(order),
            ocl::SpatialDims::from(order / 16),
            &vec![0; 0],
        ),
        (
            "OpenCL, matrix mult, C row, A row in priv mem",
            "../C_row_priv.cl",
            ocl::SpatialDims::from(order),
            ocl::SpatialDims::from(order / 16),
            &vec![0; 0],
        ),
        (
            "OpenCL, mat mult, C row, priv A, B cols loc",
            "../C_row_priv_bloc.cl",
            ocl::SpatialDims::from(order),
            ocl::SpatialDims::from(order / 16),
            &vec![order],
        ),
        (
            "Parallel matrix mult (blocked)",
            "../C_block_form.cl",
            ocl::SpatialDims::from((order, order)),
            ocl::SpatialDims::from((blocksize, blocksize)),
            &vec![blocksize * blocksize; 2],
        ),
    ].iter()
    {
        println!("===== {}, order {} =====", title, order);
        let mut kernel_src = String::new();
        std::fs::File::open(path)?.read_to_string(&mut kernel_src)?;
        let proque = ocl::ProQue::builder()
            .src(kernel_src.clone())
            .dims(order * order)
            .build()?;

        let d_c = proque
            .buffer_builder()
            .flags(ocl::flags::MemFlags::new().write_only())
            .build()?;
        let mut h_c = vec![0.0; order * order];

        let kernel = local_mem
            .iter()
            .fold(
                proque
            .kernel_builder("mmul")
            .global_work_size(global_size)
            .local_work_size(local_size)
            .arg(ocl::prm::Int::new(order as ocl::prm::cl_int))
            .arg(&proque.buffer_builder().fill_val(AVAL).build()?)  // d_a
            .arg(&proque.buffer_builder().fill_val(BVAL).build()?)  // d_b
            .arg(&d_c),
                |kernel_builder, local_mem_size| {
                    kernel_builder.arg_local::<ocl::prm::cl_float>(*local_mem_size)
                },
            )
            .build()?;

        for _i in 0..COUNT {
            let start_time = std::time::Instant::now();
            unsafe {
                kernel.enq().expect("Failed to execute OpenCL kernel");
            }
            let run_time_ns = start_time.elapsed().as_secs() * 1000000000
                + (start_time.elapsed().subsec_nanos() as u64);
            d_c.read(&mut h_c).enq()?;
            println!(
                "{} seconds at {} MFLOPS", // GFLOPS?
                (run_time_ns as f64) / 1000000000.0,
                (2 * (order as u64) * (order as u64) * (order as u64)) as f64
                    / (run_time_ns as f64)
            );
            // error checking needed
        }
    }
    Ok(())
}

fn main() {
    post_rustbook().unwrap()
}
