use crate::raw::{cl_bool, cl_uint, cl_ulong, CL_FALSE};
use crate::{Error, Result};
use generic_array::{ArrayLength, GenericArray};
use libc::size_t;
use sealed::OclInfoInternal;
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

/// A trait implemented by OpenCL wrapper types to provide access to OpenCL
/// information functions
pub trait OclInfo: sealed::OclInfoInternal {
    /// Get raw binary info from OpenCL about this object.
    ///
    /// This function performs two calls to the underlying `clGet___Info`
    /// function - one to determine the size of the information, and one to read
    /// the data once an appropriately-sized vector has been allocated to store
    /// it. If the reported size of the data changes between the two calls,
    /// `Error::InvalidDataLength` will be returned.
    ///
    /// If the size of the data is known at compile time, `get_info_raw_sized`
    /// should be preferred, as it only requires one call to `clGet___Info` and
    /// does not perform any heap allocations.
    fn get_info_raw(&self, param_name: Self::Param) -> Result<Vec<u8>> {
        unsafe {
            let mut size = 0;

            wrap_result!(Self::DEBUG_CONTEXT => self.raw_info_internal(
                param_name,
                0,
                null_mut(),
                &mut size as _
            ))?;

            let mut data = vec![0u8; size as usize];

            wrap_result!(Self::DEBUG_CONTEXT => self.raw_info_internal(
                param_name,
                size,
                data.as_mut_ptr() as *mut _,
                &mut size as _
            ))?;

            if data.len() != size {
                return Err(Error::InvalidDataLength {
                    expected: data.len(),
                    actual: size,
                });
            }

            Ok(data)
        }
    }

    /// Get raw binary info from OpenCL about this object, with a constant size.
    ///
    /// If the size of the data as reported by OpenCL doesn't match the expected
    /// size as specified by the generic parameter, `Error::InvalidLength` will
    /// be returned.
    ///
    /// If the size of the data isn't known at compile time, `get_info_raw` can
    /// be used instead.
    fn get_info_raw_sized<L: ArrayLength<u8>>(
        &self,
        param_name: Self::Param,
    ) -> Result<GenericArray<u8, L>> {
        unsafe {
            let mut array = GenericArray::default();
            let mut size_ret = 0;

            wrap_result!(Self::DEBUG_CONTEXT => self.raw_info_internal(
                param_name,
                L::USIZE,
                array.as_mut_ptr() as _,
                &mut size_ret as _
            ))?;

            if L::USIZE != size_ret {
                return Err(Error::InvalidDataLength {
                    expected: L::USIZE,
                    actual: size_ret,
                });
            }

            Ok(array)
        }
    }

    /// Get information about this object from OpenCL.
    ///
    /// This function will automatically convert the data to the type specified
    /// by the type parameter, but it's up to the programmer to ensure that this
    /// is the appropriate type for the given parameter.
    fn get_info<T: FromOclInfo>(&self, param_name: Self::Param) -> Result<T>
    where
        Self: Sized,
    {
        T::read(self, param_name)
    }
}

impl<T: sealed::OclInfoInternal> OclInfo for T {}

/// A trait to get OpenCL information and automatically convert it to a more
/// useful type.
pub trait FromOclInfo: Sized {
    fn read<T: OclInfo>(from: &T, param_name: T::Param) -> Result<Self>;
}

impl FromOclInfo for Vec<u8> {
    fn read<T: OclInfo>(from: &T, param_name: <T as OclInfoInternal>::Param) -> Result<Self> {
        from.get_info_raw(param_name)
    }
}

impl FromOclInfo for CString {
    fn read<T: OclInfo>(from: &T, param_name: T::Param) -> Result<Self> {
        let mut data = from.get_info_raw(param_name)?;

        if let Some(i) = data.iter().copied().position(|b| b == b'\0') {
            data.truncate(i);
        }

        Ok(CString::new(data).unwrap())
    }
}

impl FromOclInfo for cl_ulong {
    fn read<T: OclInfo>(from: &T, param_name: T::Param) -> Result<Self> {
        from.get_info_raw_sized(param_name)
            .map(|d| Self::from_ne_bytes(d.into()))
    }
}

impl FromOclInfo for size_t {
    fn read<T: OclInfo>(from: &T, param_name: T::Param) -> Result<Self> {
        from.get_info_raw_sized(param_name)
            .map(|d| Self::from_ne_bytes(d.into()))
    }
}

impl FromOclInfo for cl_uint {
    fn read<T: OclInfo>(from: &T, param_name: T::Param) -> Result<Self> {
        from.get_info_raw_sized(param_name)
            .map(|d| Self::from_ne_bytes(d.into()))
    }
}

impl FromOclInfo for bool {
    fn read<T: OclInfo>(from: &T, param_name: T::Param) -> Result<Self> {
        from.get_info_raw_sized(param_name)
            .map(|d| cl_bool::from_ne_bytes(d.into()) != CL_FALSE)
    }
}

impl FromOclInfo for Vec<size_t> {
    fn read<T: OclInfo>(from: &T, param_name: T::Param) -> Result<Self> {
        let raw = from.get_info_raw(param_name)?;
        raw.chunks(size_of::<size_t>())
            .map(|c| {
                c.try_into()
                    .map(size_t::from_ne_bytes)
                    .map_err(|_| Error::InvalidDataLength {
                        expected: size_of::<size_t>(),
                        actual: c.len(),
                    })
            })
            .collect()
    }
}

impl<P> FromOclInfo for *mut P {
    fn read<T: OclInfo>(from: &T, param_name: T::Param) -> Result<Self> {
        size_t::read(from, param_name).map(|p| p as _)
    }
}

impl<P> FromOclInfo for Vec<*mut P> {
    fn read<T: OclInfo>(from: &T, param_name: T::Param) -> Result<Self> {
        Ok(Vec::<size_t>::read(from, param_name)?
            .into_iter()
            .map(|p| p as _)
            .collect())
    }
}
