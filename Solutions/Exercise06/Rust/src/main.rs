extern crate ocl;

extern crate float_cmp;

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

    println!(
        "===== OpenCL, matrix mult, C(i,j) per work item, order {} =====",
        order
    );
    let proque = ocl::ProQue::builder()
        .src(
            "__kernel void mmul(
               const int N,
                __global float* A,
                __global float* B,
                __global float* C)
              {
                int k;
                int i = get_global_id(0);
                int j = get_global_id(1);
                float tmp = 0;
                if ((i < N) && (j < N))
                {
                    tmp = 0.0f;
                    for (k=0; k<N; k++)
                    {
                        tmp += A[i*N + k] * B[k*N + j];
                    }
                    C[i*N + j] = tmp;
                }
              }",
        )
        .dims(order * order)
        .build()?;
    println!("{:?}", proque.dims());

    let d_c = proque
        .buffer_builder()
        .flags(ocl::flags::MemFlags::new().write_only())
        .build()?;
    let mut h_c = vec![0.0; order * order];

    let kernel = proque
        .kernel_builder("mmul")
        .arg(ocl::prm::Int::new(order as ocl::prm::cl_int))
        .arg(&proque.buffer_builder().fill_val(AVAL).build()?)  // d_a
        .arg(&proque.buffer_builder().fill_val(BVAL).build()?)  // d_b
        .arg(&d_c)
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
            "{} seconds at {} MFLOPS",
            (run_time_ns as f64) / 1000000000.0,
            (2 * (order as u64) * (order as u64) * (order as u64)) as f64 / (run_time_ns as f64)
        );
        // error checking needed
    }
    Ok(())
}

fn main() {
    post_rustbook().unwrap()
}
