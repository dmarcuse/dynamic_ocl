//! Buffer access and use flags, to allow validity checks to be performed at
//! compile time by the type system.

use crate::raw::{
    cl_mem_flags, CL_MEM_ALLOC_HOST_PTR, CL_MEM_HOST_NO_ACCESS, CL_MEM_HOST_READ_ONLY,
    CL_MEM_HOST_WRITE_ONLY, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY,
};

mod sealed {
    use crate::raw::cl_mem_flags;

    pub trait FlagInternal {
        const FLAGS: cl_mem_flags;
    }
}

/// A trait denoting a buffer host accessibility type.
///
/// Types implementing this trait indicate whether an OpenCL memory object can
/// be read/written by the host.
pub trait HostAccess: sealed::FlagInternal {}

/// A trait denoting a buffer that may be read by the host.
pub trait HostReadable: HostAccess {}

/// A trait denoting a buffer that may be written by the host.
pub trait HostWritable: HostAccess {}

/// The host may not read or write the buffer once it's been created.
pub struct HostNoAccess;

/// The host may only read the buffer once it's been created.
pub struct HostReadOnly;

/// The host may only write the buffer once it's been created.
pub struct HostWriteOnly;

/// The host may read and write the buffer.
pub struct HostReadWrite;

impl sealed::FlagInternal for HostNoAccess {
    const FLAGS: cl_mem_flags = CL_MEM_HOST_NO_ACCESS;
}

impl sealed::FlagInternal for HostReadOnly {
    const FLAGS: cl_mem_flags = CL_MEM_HOST_READ_ONLY;
}

impl sealed::FlagInternal for HostWriteOnly {
    const FLAGS: cl_mem_flags = CL_MEM_HOST_WRITE_ONLY;
}

impl sealed::FlagInternal for HostReadWrite {
    const FLAGS: cl_mem_flags = 0;
}

impl HostAccess for HostNoAccess {}

impl HostAccess for HostReadOnly {}
impl HostReadable for HostReadOnly {}

impl HostAccess for HostWriteOnly {}
impl HostWritable for HostWriteOnly {}

impl HostAccess for HostReadWrite {}
impl HostReadable for HostReadWrite {}
impl HostWritable for HostReadWrite {}

/// A trait denoting a buffer device accessibility type.
///
/// Types implementing this trait indicate whether an OpenCL memory object can
/// be read/written by the OpenCL device.
pub trait DeviceAccess: sealed::FlagInternal {}

/// The device may only read the buffer.
pub struct DeviceReadOnly;

/// The device may only write the buffer.
pub struct DeviceWriteOnly;

/// The device may read and write the buffer.
pub struct DeviceReadWrite;

impl sealed::FlagInternal for DeviceReadOnly {
    const FLAGS: cl_mem_flags = CL_MEM_READ_ONLY;
}

impl sealed::FlagInternal for DeviceWriteOnly {
    const FLAGS: cl_mem_flags = CL_MEM_WRITE_ONLY;
}

impl sealed::FlagInternal for DeviceReadWrite {
    const FLAGS: cl_mem_flags = CL_MEM_READ_WRITE;
}

impl DeviceAccess for DeviceReadOnly {}
impl DeviceAccess for DeviceWriteOnly {}
impl DeviceAccess for DeviceReadWrite {}

/// A trait used to specify extra buffer flags.
pub trait BufferFlags: sealed::FlagInternal {}

/// Don't set any special buffer flags.
pub struct NoFlags;

/// Set the `CL_MEM_ALLOC_HOST_PTR` flag indicating that the buffer should be
/// allocated in host-accessible memory.
pub struct AllocHostPtr;

impl sealed::FlagInternal for NoFlags {
    const FLAGS: cl_mem_flags = 0;
}

impl sealed::FlagInternal for AllocHostPtr {
    const FLAGS: cl_mem_flags = CL_MEM_ALLOC_HOST_PTR;
}

impl BufferFlags for NoFlags {}
impl BufferFlags for AllocHostPtr {}
