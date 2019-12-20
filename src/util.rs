use crate::raw::cl_ulong;
use crate::Result;
use std::convert::TryInto;
use std::ffi::CString;
use std::ptr::null_mut;

pub(crate) mod sealed {
    use crate::raw::cl_int;
    use libc::size_t;
    use std::ffi::c_void;

    pub trait OclInfoInternal {
        type Param: Copy;

        const DEBUG_CONTEXT: &'static str;

        unsafe fn raw_info_internal(
            &self,
            param_name: Self::Param,
            param_value_size: size_t,
            param_value: *mut c_void,
            param_value_size_ret: *mut size_t,
        ) -> cl_int;
    }
}

pub trait OclInfo: sealed::OclInfoInternal {
    /// Get raw binary info from OpenCL about this object.
    fn get_info_raw(&self, param_name: Self::Param) -> Result<Vec<u8>> {
        unsafe {
            let mut size = 0;

            ocl_try!(Self::DEBUG_CONTEXT => self.raw_info_internal(
                param_name,
                0,
                null_mut(),
                &mut size as _
            ));

            let mut data = vec![0u8; size as usize];

            ocl_try!(Self::DEBUG_CONTEXT => self.raw_info_internal(
                param_name,
                size,
                data.as_mut_ptr() as *mut _,
                null_mut()
            ));

            Ok(data)
        }
    }

    fn get_info_string(&self, param_name: Self::Param) -> Result<CString> {
        let mut bytes = self.get_info_raw(param_name)?;

        if let Some(i) = bytes.iter().position(|&b| b == b'\0') {
            bytes.truncate(i);
        }

        Ok(CString::new(bytes).unwrap())
    }

    fn get_info_ulong(&self, param_name: Self::Param) -> Result<cl_ulong> {
        Ok(cl_ulong::from_ne_bytes(
            self.get_info_raw(param_name)?
                .as_slice()
                .try_into()
                .expect("invalid info size"),
        ))
    }
}

impl<T: sealed::OclInfoInternal> OclInfo for T {}
