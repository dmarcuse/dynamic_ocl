extern crate dynamic_ocl;

use dlopen::wrapper::Container;
use dynamic_ocl::raw::OpenCL;

pub fn main() {
    let ocl: Container<OpenCL> =
        unsafe { Container::load("libOpenCL.so") }.expect("Error loading OpenCL library");

    println!("Successfully loaded OpenCL: {:#?}", *ocl);
}
