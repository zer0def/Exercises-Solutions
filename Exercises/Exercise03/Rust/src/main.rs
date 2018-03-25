extern crate ocl;

extern crate float_cmp;
extern crate rand;

use rand::Rng;
use float_cmp::ApproxEqRatio;

const TOL: f32 = 0.001;

fn post_rustbook() -> Result<(), ocl::error::Error> {
    let context = ocl::Context::builder().build()?;
    let kernel_src = "__kernel void vadd(
        __global float* a,
        __global float* b,
        __global float* c,
        const unsigned int count)
    {
        int i = get_global_id(0);
        if (i < count) c[i] = a[i] + b[i];
    }";

    let dev_idx = 0;
    let device = context.get_device_by_wrapping_index(dev_idx);
    println!("Device is {}", device.name()?);
    println!(
        "{} from {} with a max of {} compute units",
        match context.device_info(dev_idx, ocl::enums::DeviceInfo::Type)? {
            ocl::enums::DeviceInfoResult::Type(value) => match value {
                ocl::flags::DeviceType::CPU => "CPU",
                ocl::flags::DeviceType::GPU => "GPU",
                _ => "unknown unit type",
            },
            _ => panic!(""),
        },
        device.vendor()?,
        match context.device_info(dev_idx, ocl::enums::DeviceInfo::MaxComputeUnits)? {
            ocl::enums::DeviceInfoResult::MaxComputeUnits(value) => value,
            _ => panic!(""),
        }
    );

    let size = match context.device_info(dev_idx, ocl::enums::DeviceInfo::MaxWorkGroupSize)? {
        ocl::enums::DeviceInfoResult::MaxWorkGroupSize(value) => value,
        _ => panic!(""),
    } as usize;

    /*    
    let queue = ocl::Queue::new(&context, device, None)?;
    let program = ocl::Program::builder()
*/
    let proque = ocl::ProQue::builder()
        // ordinarily you don't need to explicitly create and assign
        // a context; added only for brevity and parity
        .context(context)
        .src(kernel_src)
        .dims(size)
//        .build(&context)
        .build()?;

    let generate_data = || {
        rand::thread_rng()
            .gen_iter::<ocl::prm::cl_float>()
            .take(size)
            .collect::<Vec<_>>()
    };
    let h_a = generate_data();
    let h_b = generate_data();
    let mut h_c = vec![0.0 as ocl::prm::cl_float; size];

    let build_read_buffer = |data| {
        ocl::Buffer::<ocl::prm::cl_float>::builder()
//            .queue(queue.clone())
            .queue(proque.queue().clone())
            .len(size)
            .flags(ocl::flags::MemFlags::new().read_only().copy_host_ptr())
            .host_data(data)
            .build().expect("Failed to create buffer")
    };
    let d_a = build_read_buffer(&h_a);
    let d_b = build_read_buffer(&h_b);

    let build_write_buffer = || {
        ocl::Buffer::<ocl::prm::cl_float>::builder()
//            .queue(queue.clone())
            .queue(proque.queue().clone())
            .len(size)
            .flags(ocl::flags::MemFlags::new().write_only())
            .build().expect("Failed to create destination buffer")
    };
    let d_c = build_write_buffer();

    /*
    let vadd = ocl::Kernel::new("vadd", &program)?
        .queue(queue.clone())
        .gws(size)
*/
    let vadd = proque
        .create_kernel("vadd")?
        .arg_buf(&d_a)
        .arg_buf(&d_b)
        .arg_buf(&d_c)
        .arg_scl(ocl::prm::Uint::new(size as ocl::prm::cl_uint));

    unsafe {
        vadd.enq().expect("Failed to execute OpenCL kernel");
    }
    d_c.read(&mut h_c).enq()?;

    let mut correct = 0;
    for (idx, result) in h_c.iter().enumerate() {
        let good_sum = h_a[idx] + h_b[idx];
        if result.approx_eq_ratio(&good_sum, TOL) {
            correct += 1
        } else {
            println!(
                "deviation: {}, h_a: {}, h_b: {}, h_c: {}",
                result - good_sum,
                h_a[idx],
                h_b[idx],
                result
            );
        };
    }
    println!("C = A+B: {} out of {} results were correct.", correct, size);
    Ok(())
}

fn main() {
    post_rustbook().unwrap()
}
