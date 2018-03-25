extern crate ocl;

fn post_rustbook() -> Result<(), ocl::error::Error> {
    let platforms = ocl::Platform::list();
    println!("Number of OpenCL platforms: {}", platforms.len());
    for p in platforms {
        let devices = ocl::Device::list_all(p)?;
        println!("-------------------------");
        println!("Platform: {}", p.name()?);
        println!("Vendor: {}", p.vendor()?);
        println!("Version: {}", p.version()?);
        println!("Number of devices: {}", devices.len());
        for d in devices {
            println!("  -------------------------");
            println!("  Name: {}", d.name()?);
            println!("  Version: {}", d.version()?);
            println!(
                "  Max. Compute Units: {}",
                match d.info(ocl::enums::DeviceInfo::MaxComputeUnits)? {
                    ocl::enums::DeviceInfoResult::MaxComputeUnits(value) => value,
                    _ => panic!(""),
                }
            );
            println!(
                "  Local Memory Size: {} KB",
                match d.info(ocl::enums::DeviceInfo::LocalMemSize)? {
                    ocl::enums::DeviceInfoResult::LocalMemSize(value) => value,
                    _ => panic!(""),
                } / (1 << 10)
            );
            println!(
                "  Global Memory Size: {} MB",
                match d.info(ocl::enums::DeviceInfo::GlobalMemSize)? {
                    ocl::enums::DeviceInfoResult::GlobalMemSize(value) => value,
                    _ => panic!(""),
                } / (1 << 20)
            );
            println!(
                "  Max Alloc Size: {} MB",
                match d.info(ocl::enums::DeviceInfo::MaxMemAllocSize)? {
                    ocl::enums::DeviceInfoResult::MaxMemAllocSize(value) => value,
                    _ => panic!(""),
                } / (1 << 20)
            );
            println!(
                "  Max Work-group Total Size: {}",
                match d.info(ocl::enums::DeviceInfo::MaxWorkGroupSize)? {
                    ocl::enums::DeviceInfoResult::MaxWorkGroupSize(value) => value,
                    _ => panic!(""),
                }
            );
            println!(
                "  Max Work-group Dimensions: {:?}",
                match d.info(ocl::enums::DeviceInfo::MaxWorkItemSizes)? {
                    ocl::enums::DeviceInfoResult::MaxWorkItemSizes(value) => value,
                    _ => panic!(""),
                }
            );
        }
    }
    Ok(())
}

fn main() {
    post_rustbook().unwrap()
}
