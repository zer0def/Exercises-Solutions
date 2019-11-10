extern crate ocl;

use std::borrow::Borrow;
use std::io::Read;
use std::ops::Deref;

const INSTEPS: usize = 1 << 27;
const NITERS: usize = 1 << 18;

fn post_rustbook() -> Result<(), ocl::error::Error> {
    let mut kernel_src = String::new();
    std::fs::File::open("../pi_ocl.cl")?.read_to_string(&mut kernel_src)?;
    let mut proque = ocl::ProQue::builder().src(kernel_src.clone()).build()?;

    let mut work_group_size = proque.max_wg_size()?;
    let mut work_groups = INSTEPS / (work_group_size * NITERS);

    if work_groups < 1 {
        work_groups = match proque
            .device()
            .info(ocl::enums::DeviceInfo::MaxComputeUnits)?
        {
            ocl::enums::DeviceInfoResult::MaxComputeUnits(value) => value as usize,
            _ => panic!(""),
        };
        work_group_size = INSTEPS / (work_groups as usize * NITERS);
    }

    let steps = work_group_size * NITERS * work_groups;
    let step_size = 1.0 / (steps as ocl::prm::cl_float);

    proque.set_dims(work_groups);
    let d_partial_sums = std::sync::Arc::new(proque
        .buffer_builder()
        .flags(ocl::flags::MemFlags::new().write_only())
        .build()?);

    println!("{} work groups of size {}", work_groups, work_group_size);
    println!("{} integration steps", steps);

    let kernel = proque
        .kernel_builder("pi")
        .global_work_size(work_groups * work_group_size)
        .local_work_size(work_group_size)
        .arg(ocl::prm::Int::new(NITERS as ocl::prm::cl_int))
        .arg(ocl::prm::Float::new(step_size as ocl::prm::cl_float))
        .arg_local::<ocl::prm::cl_float>(work_group_size)
        .arg(d_partial_sums.deref())
        .build()?;

    println!("{:?}", kernel.default_global_work_size());
    println!("{:?}", kernel.default_local_work_size());

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
    post_rustbook().unwrap()
}
