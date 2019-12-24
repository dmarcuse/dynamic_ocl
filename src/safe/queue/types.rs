use crate::raw::*;
use crate::safe::context::Context;
use crate::safe::device::Device;
use crate::safe::queue::Queue;
use crate::Result;

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
