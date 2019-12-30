extern crate dynamic_ocl;

use dynamic_ocl::device::DeviceType;
use dynamic_ocl::load_opencl;
use dynamic_ocl::platform::Platform;
use dynamic_ocl::program::ProgramBuilder;
use dynamic_ocl::queue::QueueBuilder;

const KERNEL: &str = r#"
__kernel void sum(__constant float *a, __constant float *b, __global float *c) {
    size_t id = get_global_id(0);
    c[id] = a[id] + b[id];
}
"#;

pub fn main() {
    let version = load_opencl().unwrap();
    println!("Successfully loaded OpenCL (compat level {:?})", version);

    for platform in Platform::get_platforms().unwrap() {
        println!("Got platform {:#?}", platform);

        for device in platform.get_devices(DeviceType::ALL).unwrap() {
            println!("Got device: {:#?}", device);

            let ctx = device.create_context().unwrap();

            println!("Created context: {:#?}", ctx);

            let queue = QueueBuilder::new(&ctx, &device).build().unwrap();

            println!("Created command queue: {:#?}", queue);

            let program = ProgramBuilder::with_source(&ctx, &KERNEL).build().unwrap();

            println!(
                "Compiled program: {:?} {:?}",
                program,
                program.kernel_names()
            );
        }
    }
}
