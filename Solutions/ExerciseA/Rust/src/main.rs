extern crate ocl;

use std::io::Read;

const INSTEPS: usize = 1 << 27;
const NITERS: usize = 1 << 18;

fn post_rustbook(vector_size: usize) -> Result<(), ocl::error::Error> {
    let (iterations, mut work_group_size, kernel_name) = match vector_size {
        1 => (
            NITERS / vector_size,
            8 * vector_size,
            std::string::String::from("pi"),
        ),
        4 | 8 => (
            NITERS / vector_size,
            8 * vector_size,
            format!("pi_vec{}", vector_size),
        ),
        _ => panic!("Invalid vector size"),
    };

    let mut kernel_src = String::new();
    std::fs::File::open("../pi_vocl.cl")?.read_to_string(&mut kernel_src)?;
    let mut proque = ocl::ProQue::builder().src(kernel_src.clone()).build()?;

    let mut work_groups = INSTEPS / (work_group_size * iterations);

    if proque.max_wg_size()? > work_group_size {
        work_group_size = proque.max_wg_size()?;
        work_groups = INSTEPS / (work_group_size * iterations);
    }

    if work_groups < 1 {
        work_groups = match proque
            .device()
            .info(ocl::enums::DeviceInfo::MaxComputeUnits)?
        {
            ocl::enums::DeviceInfoResult::MaxComputeUnits(value) => value as usize,
            _ => panic!(""),
        };
        work_group_size = INSTEPS / (work_groups * iterations);
    }

    let steps = work_group_size * iterations * work_groups;
    let step_size = 1.0 / (steps as ocl::prm::cl_float);

    proque.set_dims(work_groups);
    let d_partial_sums = proque
        .buffer_builder()
        .flags(ocl::flags::MemFlags::new().write_only())
        .build()?;

    println!("{} work groups of size {}", work_groups, work_group_size);
    println!("{} integration steps", steps);
    let start_time = std::time::Instant::now();

    let kernel = proque
        .kernel_builder(kernel_name.as_str())
        .global_work_size(work_groups * work_group_size)
        .local_work_size(work_group_size)
        .arg(ocl::prm::Int::new(iterations as ocl::prm::cl_int))
        .arg(ocl::prm::Float::new(step_size as ocl::prm::cl_float))
        .arg_local::<ocl::prm::cl_float>(work_group_size)
        .arg(&d_partial_sums)
        .build()?;

    let start_time = std::time::Instant::now();
    unsafe {
        kernel.enq().expect("Failed to execute OpenCL kernel");
    }
    let run_time_ns =
        start_time.elapsed().as_secs() * 1000000000 + (start_time.elapsed().subsec_nanos() as u64);

    let mut h_psum = vec![0.0; work_groups];
    d_partial_sums.read(&mut h_psum).enq()?;
    let pi_res: ocl::prm::cl_float = h_psum.iter().sum::<ocl::prm::cl_float>() * step_size;

    println!(
        "The calculation ran in {} seconds",
        (run_time_ns as f64) / 1000000000.0
    );
    println!("pi = {} for {} steps", pi_res, steps);

    Ok(())
}

fn main() {
    match std::env::args().nth(1) {
        Some(value) => post_rustbook(value.parse().unwrap()).unwrap(),
        _ => println!(
            "Usage: `{} <num>` where <num> = 1, 4 or 8",
            std::env::args().nth(0).unwrap()
        ),
    }
}
