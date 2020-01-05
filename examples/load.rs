extern crate dynamic_ocl;

use dynamic_ocl::buffer::flags::{DeviceReadOnly, DeviceWriteOnly, HostNoAccess, HostReadOnly};
use dynamic_ocl::device::DeviceType;
use dynamic_ocl::load_opencl;
use dynamic_ocl::platform::Platform;
use dynamic_ocl::program::ProgramBuilder;
use dynamic_ocl::queue::QueueBuilder;
use std::ffi::CString;

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

            let a = ctx
                .buffer_builder()
                .host_access::<HostNoAccess>()
                .device_access::<DeviceReadOnly>()
                .build_copying_slice(&[1.0f32, 2.0, 3.0])
                .unwrap();

            let b = ctx
                .buffer_builder()
                .host_access::<HostNoAccess>()
                .device_access::<DeviceReadOnly>()
                .build_copying_slice(&[1.0f32, 2.0, 3.0])
                .unwrap();

            let c = ctx
                .buffer_builder()
                .host_access::<HostReadOnly>()
                .device_access::<DeviceWriteOnly>()
                .build_with_size::<f32>(3)
                .unwrap();

            let args = (a, b, c);

            println!("Created arguments: {:#?}", args);

            let kernel = program
                .create_kernel(&CString::new("sum").unwrap())
                .unwrap()
                .bind_arguments(args)
                .unwrap();

            println!("Created and bound kernel: {:#?}", kernel);
        }
    }
}
