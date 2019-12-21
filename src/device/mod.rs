mod types;

use crate::context::Context;
use crate::platform::Platform;
use crate::queue::QueueProperties;
use crate::raw::{
    cl_context_properties, cl_device_id, cl_device_info, cl_platform_id, cl_uint, cl_ulong,
    CL_CONTEXT_PLATFORM, CL_SUCCESS,
};
use crate::util::sealed::OclInfoInternal;
use crate::Result;
use libc::size_t;
use std::ffi::CString;
use std::fmt::{self, Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::os::raw::c_void;
use std::ptr::null_mut;
pub use types::*;

/// An OpenCL device
#[derive(Clone)]
pub struct Device {
    pub(crate) ocl: crate::OpenCL,
    pub(crate) id: cl_device_id,
}

impl Debug for Device {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.info_fmt(f)
    }
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Device {}

impl Hash for Device {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.id as usize)
    }
}

impl OclInfoInternal for Device {
    type Param = cl_device_info;
    const DEBUG_CONTEXT: &'static str = "clGetDeviceInfo";

    unsafe fn raw_info_internal(
        &self,
        param_name: Self::Param,
        param_value_size: usize,
        param_value: *mut c_void,
        param_value_size_ret: *mut usize,
    ) -> i32 {
        self.ocl.raw().CL10.clGetDeviceInfo(
            self.id,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret,
        )
    }
}

impl Device {
    /// Create a new context containing only this device, with no custom
    /// properties set.
    pub fn create_context(&self) -> Result<Context> {
        unsafe {
            let mut props = [CL_CONTEXT_PLATFORM, self.platform_id()? as _, 0];
            let mut err = CL_SUCCESS;
            let id = self.ocl.raw().CL10.clCreateContext(
                props.as_mut_ptr(),
                1,
                &self.id as *const _,
                None,
                null_mut(),
                &mut err as _,
            );
            ocl_try!("clCreateContext" => err);
            Ok(Context {
                id,
                ocl: self.ocl.clone(),
            })
        }
    }

    info_funcs! {
        pub fn device_type(&self) -> DeviceType = self.get_info_ulong(CL_DEVICE_TYPE);
        pub fn device_vendor_id(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_VENDOR_ID);
        pub fn max_compute_units(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_MAX_COMPUTE_UNITS);
        pub fn max_work_item_dimensions(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
        pub fn max_work_item_sizes(&self) -> Vec<size_t> = self.get_info_raw(CL_DEVICE_MAX_WORK_ITEM_SIZES);
        pub fn max_work_group_size(&self) -> size_t = self.get_info_size_t(CL_DEVICE_MAX_WORK_GROUP_SIZE);
        pub fn preferred_vector_width_char(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR);
        pub fn preferred_vector_width_short(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT);
        pub fn preferred_vector_width_int(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT);
        pub fn preferred_vector_width_long(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG);
        pub fn preferred_vector_width_float(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT);
        pub fn preferred_vector_width_double(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE);
        pub fn preferred_vector_width_half(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF);
        pub fn native_vector_width_char(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR);
        pub fn native_vector_width_short(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT);
        pub fn native_vector_width_int(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_NATIVE_VECTOR_WIDTH_INT);
        pub fn native_vector_width_long(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG);
        pub fn native_vector_width_float(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT);
        pub fn native_vector_width_double(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE);
        pub fn native_vector_width_half(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF);
        pub fn max_clock_frequency(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_MAX_CLOCK_FREQUENCY);
        pub fn address_bits(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_ADDRESS_BITS);
        pub fn max_mem_alloc_size(&self) -> cl_ulong = self.get_info_ulong(CL_DEVICE_MAX_MEM_ALLOC_SIZE);
        pub fn image_support(&self) -> bool = self.get_info_bool(CL_DEVICE_IMAGE_SUPPORT);
        pub fn max_read_image_args(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_MAX_READ_IMAGE_ARGS);
        pub fn max_write_image_args(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_MAX_WRITE_IMAGE_ARGS);
        pub fn max_read_write_image_args(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS);
        pub fn il_version(&self) -> CString = self.get_info_string(CL_DEVICE_IL_VERSION);
        pub fn image2d_max_width(&self) -> size_t = self.get_info_size_t(CL_DEVICE_IMAGE2D_MAX_WIDTH);
        pub fn image2d_max_height(&self) -> size_t = self.get_info_size_t(CL_DEVICE_IMAGE2D_MAX_HEIGHT);
        pub fn image3d_max_width(&self) -> size_t = self.get_info_size_t(CL_DEVICE_IMAGE3D_MAX_WIDTH);
        pub fn image3d_max_height(&self) -> size_t = self.get_info_size_t(CL_DEVICE_IMAGE3D_MAX_HEIGHT);
        pub fn image3d_max_depth(&self) -> size_t = self.get_info_size_t(CL_DEVICE_IMAGE3D_MAX_DEPTH);
        pub fn image_max_buffer_size(&self) -> size_t = self.get_info_size_t(CL_DEVICE_IMAGE_MAX_BUFFER_SIZE);
        pub fn image_max_array_size(&self) -> size_t = self.get_info_size_t(CL_DEVICE_IMAGE_MAX_ARRAY_SIZE);
        pub fn max_samplers(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_MAX_SAMPLERS);
        pub fn image_pitch_alignment(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_IMAGE_PITCH_ALIGNMENT);
        pub fn image_base_address_alignment(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT);
        pub fn max_pipe_args(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_MAX_PIPE_ARGS);
        pub fn pipe_max_active_reservations(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS);
        pub fn pipe_max_packet_size(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_PIPE_MAX_PACKET_SIZE);
        pub fn max_parameter_size(&self) -> size_t = self.get_info_size_t(CL_DEVICE_MAX_PARAMETER_SIZE);
        pub fn mem_base_addr_align(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_MEM_BASE_ADDR_ALIGN);
        pub fn single_fp_config(&self) -> FPConfig = self.get_info_ulong(CL_DEVICE_SINGLE_FP_CONFIG);
        pub fn double_fp_config(&self) -> FPConfig = self.get_info_ulong(CL_DEVICE_DOUBLE_FP_CONFIG);
        pub fn global_mem_cache_type(&self) -> MemCacheType = self.get_info_uint(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE);
        pub fn global_mem_cacheline_size(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE);
        pub fn global_mem_cache_size(&self) -> cl_ulong = self.get_info_ulong(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE);
        pub fn global_mem_size(&self) -> cl_ulong = self.get_info_ulong(CL_DEVICE_GLOBAL_MEM_SIZE);
        pub fn max_constant_buffer_size(&self) -> cl_ulong = self.get_info_ulong(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE);
        pub fn max_constant_args(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_MAX_CONSTANT_ARGS);
        pub fn max_global_variable_size(&self) -> size_t = self.get_info_size_t(CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE);
        pub fn global_variable_preferred_total_size(&self) -> size_t = self.get_info_size_t(CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE);
        pub fn local_mem_type(&self) -> LocalMemType = self.get_info_uint(CL_DEVICE_LOCAL_MEM_TYPE);
        pub fn local_mem_size(&self) -> cl_ulong = self.get_info_ulong(CL_DEVICE_LOCAL_MEM_SIZE);
        pub fn error_correction_support(&self) -> bool = self.get_info_bool(CL_DEVICE_ERROR_CORRECTION_SUPPORT);
        pub fn profiling_timer_resolution(&self) -> size_t = self.get_info_size_t(CL_DEVICE_PROFILING_TIMER_RESOLUTION);
        pub fn endian_little(&self) -> bool = self.get_info_bool(CL_DEVICE_ENDIAN_LITTLE);
        pub fn available(&self) -> bool = self.get_info_bool(CL_DEVICE_AVAILABLE);
        pub fn linker_available(&self) -> bool = self.get_info_bool(CL_DEVICE_LINKER_AVAILABLE);
        pub fn execution_capabilities(&self) -> ExecutionCapabilities = self.get_info_ulong(CL_DEVICE_EXECUTION_CAPABILITIES);
        pub fn queue_on_host_properties(&self) -> QueueProperties = self.get_info_ulong(CL_DEVICE_QUEUE_ON_HOST_PROPERTIES);
        pub fn queue_on_device_properties(&self) -> QueueProperties = self.get_info_ulong(CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES);
        pub fn queue_on_device_preferred_size(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE);
        pub fn queue_on_device_max_size(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE);
        pub fn max_on_device_queues(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_MAX_ON_DEVICE_QUEUES);
        pub fn max_on_device_events(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_MAX_ON_DEVICE_EVENTS);
        pub fn built_in_kernels(&self) -> CString = self.get_info_string(CL_DEVICE_BUILT_IN_KERNELS);
        pub fn platform_id(&self) -> cl_platform_id = self.get_info_size_t(CL_DEVICE_PLATFORM);
        pub fn name(&self) -> CString = self.get_info_string(CL_DEVICE_NAME);
        pub fn vendor(&self) -> CString = self.get_info_string(CL_DEVICE_VENDOR);
        pub fn driver_version(&self) -> CString = self.get_info_string(CL_DRIVER_VERSION);
        pub fn profile(&self) -> CString = self.get_info_string(CL_DEVICE_PROFILE);
        pub fn version(&self) -> CString = self.get_info_string(CL_DEVICE_VERSION);
        pub fn opencl_c_version(&self) -> CString = self.get_info_string(CL_DEVICE_OPENCL_C_VERSION);
        pub fn extensions(&self) -> CString = self.get_info_string(CL_DEVICE_EXTENSIONS);
        pub fn printf_buffer_size(&self) -> size_t = self.get_info_size_t(CL_DEVICE_PRINTF_BUFFER_SIZE);
        pub fn preferred_interop_user_sync(&self) -> bool = self.get_info_bool(CL_DEVICE_PREFERRED_INTEROP_USER_SYNC);
        pub fn parent_device_id(&self) -> cl_device_id = self.get_info_size_t(CL_DEVICE_PARENT_DEVICE);
        pub fn partition_max_sub_devices(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_PARTITION_MAX_SUB_DEVICES);
        // TODO: CL_DEVICE_PARTITION_PROPERTIES
        pub fn partition_affinity_domain(&self) -> AffinityDomain = self.get_info_ulong(CL_DEVICE_PARTITION_AFFINITY_DOMAIN);
        // TODO: CL_DEVICE_PARTITION_TYPE
        pub fn reference_count(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_REFERENCE_COUNT);
        pub fn svm_capabilities(&self) -> SVMCapabilities = self.get_info_ulong(CL_DEVICE_SVM_CAPABILITIES);
        pub fn preferred_platform_atomic_alignment(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT);
        pub fn preferred_global_atomic_alignment(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT);
        pub fn preferred_local_atomic_alignment(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT);
        pub fn max_num_sub_groups(&self) -> cl_uint = self.get_info_uint(CL_DEVICE_MAX_NUM_SUB_GROUPS);
        pub fn sub_group_independent_forward_progress(&self) -> bool = self.get_info_bool(CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS);
    }

    pub fn platform(&self) -> Result<Platform> {
        self.platform_id().map(|id| Platform {
            id,
            ocl: self.ocl.clone(),
        })
    }

    pub fn parent_device(&self) -> Result<Option<Device>> {
        self.parent_device_id().map(|id| {
            if id.is_null() {
                None
            } else {
                Some(Device {
                    id,
                    ocl: self.ocl.clone(),
                })
            }
        })
    }
}
