use crate::buffer::flags::HostAccess;
use crate::buffer::MemSafe;
use crate::context::Context;
use crate::device::Device;
use crate::kernel::{Kernel, KernelArgList, KernelInfo};
use crate::queue::Queue;
use crate::raw::*;
use crate::safe::buffer::flags::{HostReadable, HostWritable};
use crate::safe::buffer::AsBuffer;
use crate::Result;
use std::mem::size_of_val;
use std::ptr::{null, null_mut};

bitfield! {
    /// Special command queue properties
    pub struct QueueProperties(cl_command_queue_properties) {
        pub const OUT_OF_ORDER_EXEC_MODE_ENABLE = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
        pub const PROFILING_ENABLE = CL_QUEUE_PROFILING_ENABLE;
        pub const ON_DEVICE = CL_QUEUE_ON_DEVICE;
        pub const ON_DEVICE_DEFAULT = CL_QUEUE_ON_DEVICE_DEFAULT;
    }
}

/// A partially built command queue
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct QueueBuilder<'c, 'd> {
    context: &'c Context,
    device: &'d Device,
    properties: Option<QueueProperties>,
    size: Option<cl_uint>,
}

impl<'c, 'd> QueueBuilder<'c, 'd> {
    /// Begin building a new command queue using the given context and device
    pub fn new(context: &'c Context, device: &'d Device) -> Self {
        Self {
            context,
            device,
            properties: None,
            size: None,
        }
    }

    /// Set command queue properties
    pub fn properties(self, properties: QueueProperties) -> Self {
        Self {
            properties: Some(properties),
            ..self
        }
    }

    /// Set the size of the command queue
    pub fn size(self, size: cl_uint) -> Self {
        Self {
            size: Some(size),
            ..self
        }
    }

    /// Build the command queue, calling either `clCreateCommandQueue` or
    /// `clCreateCommandQueueWithProperties` depending on builder parameters and
    /// system OpenCL version.
    pub fn build(self) -> Result<Queue> {
        unsafe {
            let mut err = CL_SUCCESS;

            let queue = match (self.properties, self.size) {
                (Some(props), Some(size)) if props.contains(QueueProperties::ON_DEVICE) => {
                    check_ocl_version!("clCreateCommandQueueWithProperties" => CL20)?;

                    let props = [
                        CL_QUEUE_PROPERTIES as _,
                        props.raw(),
                        CL_QUEUE_SIZE as _,
                        size as _,
                        0,
                    ];

                    clCreateCommandQueueWithProperties(
                        self.context.raw(),
                        self.device.raw(),
                        props.as_ptr(),
                        &mut err as _,
                    )
                }
                (_, Some(_)) => {
                    panic!("cannot set queue size unless queue property ON_DEVICE is set")
                }
                (props, None) => clCreateCommandQueue(
                    self.context.raw(),
                    self.device.raw(),
                    props.map(|p| p.raw()).unwrap_or_default(),
                    &mut err as _,
                ),
            };

            wrap_result!("clCreateCommandQueue" => err)?;
            Ok(Queue(queue))
        }
    }
}

/// A partially built command to interact with a buffer
#[must_use]
pub struct BufferCmd<'q, 'a, H: HostAccess, T: MemSafe> {
    pub(super) queue: &'q Queue,
    pub(super) buffer: &'q mut dyn AsBuffer<'a, H, T>,
    pub(super) offset: Option<usize>,
}

impl<'q, 'a, H: HostAccess, T: MemSafe> BufferCmd<'q, 'a, H, T> {
    /// Set the offset within the OpenCL buffer for this memory operation.
    ///
    /// Offsets in host memory should be set using slicing.
    pub fn offset(self, offset: usize) -> Self {
        Self {
            offset: Some(offset),
            ..self
        }
    }

    /// Perform a blocking read of the buffer into the given slice.
    pub fn read(self, dest: &mut [T]) -> Result<()>
    where
        H: HostReadable,
    {
        unsafe {
            wrap_result!("clEnqueueReadBuffer" => clEnqueueReadBuffer(
                self.queue.raw(),
                self.buffer.as_buffer().raw(),
                CL_TRUE,
                self.offset.unwrap_or(0),
                size_of_val(dest),
                dest as *mut _ as _,
                0,
                null_mut(),
                null_mut()
            ))?;

            Ok(())
        }
    }

    /// Perform a blocking write of the buffer into the given slice.
    pub fn write(self, src: &[T]) -> Result<()>
    where
        H: HostWritable,
    {
        unsafe {
            wrap_result!("clEnqueueWriteBuffer" => clEnqueueWriteBuffer(
                self.queue.raw(),
                self.buffer.as_buffer().raw(),
                CL_TRUE,
                self.offset.unwrap_or(0),
                size_of_val(src),
                src as *const _ as _,
                0,
                null_mut(),
                null_mut(),
            ))?;

            Ok(())
        }
    }

    /// Fill the buffer with the given pattern, blocking until completion.
    pub fn fill(self, pattern: &T) -> Result<()> {
        unsafe {
            let mut event = null_mut();

            wrap_result!("clEnqueueFillBuffer" => clEnqueueFillBuffer(
                self.queue.raw(),
                self.buffer.as_buffer().raw(),
                pattern as *const _ as _,
                size_of_val(pattern),
                self.offset.unwrap_or(0),
                self.buffer.as_buffer().rust_size(),
                0,
                null_mut(),
                &mut event as _
            ))?;

            wrap_result!("clWaitForEvents" => clWaitForEvents(1, &event as _))?;

            Ok(())
        }
    }
}

/// A trait implemented for types which can be used to specify kernel work
/// sizes/offsets
pub trait WorkDims {
    const NUM_WORK_DIMS: u32;
    fn as_ptr(&self) -> *const usize;
}

impl WorkDims for usize {
    const NUM_WORK_DIMS: u32 = 1;

    fn as_ptr(&self) -> *const usize {
        self as _
    }
}

impl WorkDims for [usize; 2] {
    const NUM_WORK_DIMS: u32 = 2;

    fn as_ptr(&self) -> *const usize {
        self[..].as_ptr()
    }
}

impl WorkDims for [usize; 3] {
    const NUM_WORK_DIMS: u32 = 3;

    fn as_ptr(&self) -> *const usize {
        self[..].as_ptr()
    }
}

/// A partially built command to execute a kernel
#[must_use]
pub struct KernelCmd<'q, T: KernelArgList, W: WorkDims> {
    pub(super) queue: &'q Queue,
    pub(super) kernel: &'q Kernel<T>,
    pub(super) global_work_offset: Option<W>,
    pub(super) local_work_size: Option<W>,
}

impl<'q, T: KernelArgList, W: WorkDims> KernelCmd<'q, T, W> {
    /// Set the global work offset
    pub fn global_work_offset(self, global_work_offset: impl Into<Option<W>>) -> Self {
        Self {
            global_work_offset: global_work_offset.into(),
            ..self
        }
    }

    /// Set the local work size
    pub fn local_work_size(self, local_work_size: impl Into<Option<W>>) -> Self {
        Self {
            local_work_size: local_work_size.into(),
            ..self
        }
    }

    /// Execute this kernel with the given global work size, blocking until
    /// completion.
    pub fn exec_ndrange(self, global_work_size: W) -> Result<()> {
        unsafe {
            let mut event = null_mut();

            wrap_result!("clEnqueueNDRangeKernel" => clEnqueueNDRangeKernel(
                self.queue.raw(),
                self.kernel.as_unbound().raw(),
                W::NUM_WORK_DIMS,
                self.global_work_offset.map(|o| o.as_ptr()).unwrap_or(null()),
                global_work_size.as_ptr(),
                self.local_work_size.map(|o| o.as_ptr()).unwrap_or(null()),
                0,
                null(),
                &mut event as _
            ))?;

            wrap_result!("clWaitForEvents" => clWaitForEvents(
                1,
                &event as _
            ))?;

            Ok(())
        }
    }
}
