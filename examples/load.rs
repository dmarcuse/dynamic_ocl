extern crate dynamic_ocl;

use dynamic_ocl::device::DeviceType;
use dynamic_ocl::load_opencl;
use dynamic_ocl::platform::Platform;

pub fn main() {
    let version = load_opencl().unwrap();
    println!("Successfully loaded OpenCL (compat level {:?})", version);

    for platform in Platform::get_platforms().unwrap() {
        println!("Got platform {:#?}", platform);

        for device in platform.get_devices(DeviceType::ALL).unwrap() {
            println!("Got device: {:#?}", device);

            let ctx = device.create_context().unwrap();

            println!("Created context: {:#?}", ctx);
        }
    }
}
