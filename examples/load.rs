extern crate dynamic_ocl;

use dlopen::utils::platform_file_name;
use dlopen::wrapper::Container;
use dynamic_ocl::raw::OpenCL;

pub fn main() {
    let ocl: Container<OpenCL> = unsafe { Container::load(platform_file_name("OpenCL")) }
        .expect("Error loading OpenCL library");

    println!("Successfully loaded OpenCL: {:#?}", *ocl);
}
