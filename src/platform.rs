use crate::device::{Device, DeviceType};
use crate::raw::{
    clGetDeviceIDs, clGetPlatformIDs, clGetPlatformInfo, cl_platform_id, cl_platform_info, cl_ulong,
};
use crate::util::sealed::OclInfoInternal;
use crate::Result;
use std::ffi::c_void;
use std::ffi::CString;
use std::fmt::{self, Debug, Formatter};
use std::hash::Hash;
use std::ptr::null_mut;

/// An OpenCL platform
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Platform(pub(crate) cl_platform_id);

impl Debug for Platform {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.info_fmt(f)
    }
}

impl OclInfoInternal for Platform {
    type Param = cl_platform_info;
    const DEBUG_CONTEXT: &'static str = "clGetPlatformInfo";

    unsafe fn raw_info_internal(
        &self,
        param_name: Self::Param,
        param_value_size: usize,
        param_value: *mut c_void,
        param_value_size_ret: *mut usize,
    ) -> i32 {
        clGetPlatformInfo(
            self.0,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret,
        )
    }
}

impl Platform {
    /// Get a list of OpenCL platforms available on this system.
    ///
    /// This call requires OpenCL 1.0+.
    pub fn get_platforms() -> Result<Vec<Platform>> {
        unsafe {
            let mut num_platforms = 0u32;

            ocl_try!("clGetPlatformIDs" => clGetPlatformIDs(
                0,
                null_mut(),
                &mut num_platforms as _
            ));

            let mut ids = vec![null_mut(); num_platforms as usize];

            ocl_try!("clGetPlatformIDs" => clGetPlatformIDs(
                num_platforms,
                ids.as_mut_ptr(),
                &mut num_platforms as _,
            ));

            Ok(ids.into_iter().map(Platform).collect())
        }
    }

    pub fn get_devices(self, typ: DeviceType) -> Result<Vec<Device>> {
        unsafe {
            let mut num_devices = 0u32;

            ocl_try!("clGetDeviceIDs" => clGetDeviceIDs(
                self.0,
                typ.raw(),
                0,
                null_mut(),
                &mut num_devices as _
            ));

            let mut ids = vec![null_mut(); num_devices as usize];

            ocl_try!("clGetDeviceIDs" => clGetDeviceIDs(
                self.0,
                typ.raw(),
                num_devices,
                ids.as_mut_ptr(),
                &mut num_devices as _,
            ));

            Ok(ids.into_iter().map(Device).collect())
        }
    }

    info_funcs! {
        pub fn profile(&self) -> CString = self.get_info_string(CL_PLATFORM_PROFILE);
        pub fn version(&self) -> CString = self.get_info_string(CL_PLATFORM_VERSION);
        pub fn name(&self) -> CString = self.get_info_string(CL_PLATFORM_NAME);
        pub fn vendor(&self) -> CString = self.get_info_string(CL_PLATFORM_VENDOR);
        pub fn extensions(&self) -> CString = self.get_info_string(CL_PLATFORM_EXTENSIONS);
        pub fn host_timer_resolution(&self) -> cl_ulong = self.get_info_ulong(CL_PLATFORM_HOST_TIMER_RESOLUTION);
    }
}
