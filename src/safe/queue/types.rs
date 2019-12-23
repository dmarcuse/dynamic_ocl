use crate::raw::*;

bitfield! {
    pub struct QueueProperties(cl_command_queue_properties) {
        pub const OUT_OF_ORDER_EXEC_MODE_ENABLE = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
        pub const PROFILING_ENABLE = CL_QUEUE_PROFILING_ENABLE;
        pub const ON_DEVICE = CL_QUEUE_ON_DEVICE;
        pub const ON_DEVICE_DEFAULT = CL_QUEUE_ON_DEVICE_DEFAULT;
    }
}
