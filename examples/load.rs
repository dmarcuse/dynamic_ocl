extern crate dynamic_ocl;

use dynamic_ocl::device::DeviceType;
use dynamic_ocl::OpenCL;

pub fn main() {
    let ocl = OpenCL::load().unwrap();
    println!("Successfully loaded OpenCL: {:#?}", ocl);

    for platform in ocl.get_platforms().unwrap() {
        println!("Got platform {:#?}", platform);

        for device in platform.get_devices(DeviceType::ALL).unwrap() {
            println!("Got device: {:#?}", device);
        }
    }
}
