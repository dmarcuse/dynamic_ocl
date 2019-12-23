
use crate::raw::{cl_uint, cl_ulong, CL_FALSE};
use crate::{Error, Result};
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

pub(crate) trait OclInfoFrom<T: ?Sized>: Sized {
    fn convert(value: &T) -> Result<Self>;
}

impl<T: Sized + Clone> OclInfoFrom<T> for T {
    fn convert(value: &T) -> Result<Self> {
        Ok(value.clone())
    }
}

impl<T> OclInfoFrom<size_t> for *mut T {
    fn convert(&value: &usize) -> Result<Self> {
        Ok(value as _)
    }
}

impl<T> OclInfoFrom<[u8]> for *mut T {
    fn convert(value: &[u8]) -> Result<Self> {
        <*mut T>::convert(&size_t::convert(value)?)
    }
}

impl OclInfoFrom<[u8]> for size_t {
    fn convert(value: &[u8]) -> Result<Self> {
        Ok(size_t::from_ne_bytes(value.try_into().map_err(|_| {
            Error::InvalidDataLength {
                expected: size_of::<size_t>(),
                actual: value.len(),
            }
        })?))
    }
}

impl<T: Sized + OclInfoFrom<[u8]>> OclInfoFrom<Vec<u8>> for Vec<T> {
    fn convert(value: &Vec<u8>) -> Result<Self> {
        let mut values = vec![];
        for chunk in value.chunks(size_of::<T>()) {
            values.push(T::convert(chunk)?);
        }
        Ok(values)
    }
}

pub(crate) fn info_convert<F, T>(value: &F) -> Result<T>
where
    T: OclInfoFrom<F>,
{
    T::convert(value)
}
