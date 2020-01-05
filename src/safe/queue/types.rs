use crate::buffer::flags::HostAccess;
use crate::buffer::MemSafe;
use crate::context::Context;
use crate::device::Device;
use crate::queue::Queue;
use crate::raw::*;
use crate::safe::buffer::flags::{HostReadable, HostWritable};
use crate::safe::buffer::AsBuffer;
use crate::Result;
use std::mem::size_of_val;
use std::ptr::null_mut;

bitfield! {
    pub struct QueueProperties(cl_command_queue_properties) {
        pub const OUT_OF_ORDER_EXEC_MODE_ENABLE = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
        pub const PROFILING_ENABLE = CL_QUEUE_PROFILING_ENABLE;
        pub const ON_DEVICE = CL_QUEUE_ON_DEVICE;
        pub const ON_DEVICE_DEFAULT = CL_QUEUE_ON_DEVICE_DEFAULT;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QueueBuilder<'c, 'd> {
    context: &'c Context,
    device: &'d Device,
    properties: Option<QueueProperties>,
    size: Option<cl_uint>,
}

impl<'c, 'd> QueueBuilder<'c, 'd> {
    pub fn new(context: &'c Context, device: &'d Device) -> Self {
        Self {
            context,
            device,
            properties: None,
            size: None,
        }
    }

    pub fn properties(self, properties: QueueProperties) -> Self {
        Self {
            properties: Some(properties),
            ..self
        }
    }

    pub fn size(self, size: cl_uint) -> Self {
        Self {
            size: Some(size),
            ..self
        }
    }

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
}
