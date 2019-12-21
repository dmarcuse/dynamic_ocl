use crate::raw::{cl_uint, cl_ulong, CL_FALSE};
use crate::Result;
use libc::size_t;
use std::convert::TryInto;
use std::ffi::CString;
use std::mem::size_of;
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

    fn get_info_uint(&self, param_name: Self::Param) -> Result<cl_uint> {
        Ok(cl_uint::from_ne_bytes(
            self.get_info_raw(param_name)?
                .as_slice()
                .try_into()
                .expect("invalid info size"),
        ))
    }

    fn get_info_size_t(&self, param_name: Self::Param) -> Result<size_t> {
        Ok(size_t::from_ne_bytes(
            self.get_info_raw(param_name)?
                .as_slice()
                .try_into()
                .expect("invalid info size"),
        ))
    }

    fn get_info_bool(&self, param_name: Self::Param) -> Result<bool> {
        self.get_info_uint(param_name).map(|b| b != CL_FALSE)
    }
}

impl<T: sealed::OclInfoInternal> OclInfo for T {}

pub(crate) trait OclInfoFrom<T>: Sized {
    fn convert(value: T) -> Result<Self>;
}

impl<T: Sized> OclInfoFrom<T> for T {
    fn convert(value: T) -> Result<Self> {
        Ok(value)
    }
}

impl OclInfoFrom<Vec<u8>> for Vec<size_t> {
    fn convert(value: Vec<u8>) -> Result<Self> {
        assert_eq!(
            value.len() % size_of::<size_t>(),
            0,
            "expected data length to be a multiple of {}",
            size_of::<size_t>()
        );

        Ok(value
            .chunks_exact(size_of::<size_t>())
            .map(|v| size_t::from_ne_bytes(v.try_into().unwrap()))
            .collect())
    }
}

impl<T> OclInfoFrom<size_t> for *mut T {
    fn convert(value: usize) -> Result<Self> {
        Ok(value as _)
    }
}

pub(crate) fn info_convert<F, T>(value: F) -> Result<T>
where
    T: OclInfoFrom<F>,
{
    T::convert(value)
}
