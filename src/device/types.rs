use crate::raw::*;

bitfield! {
    /// A bitfield representing OpenCL device types
    pub struct DeviceType(cl_device_type) {
        pub const ALL = CL_DEVICE_TYPE_ALL;
        pub const CPU = CL_DEVICE_TYPE_CPU;
        pub const GPU = CL_DEVICE_TYPE_GPU;
        pub const DEFAULT = CL_DEVICE_TYPE_DEFAULT;
        pub const ACCELERATOR = CL_DEVICE_TYPE_ACCELERATOR;
        pub const CUSTOM = CL_DEVICE_TYPE_CUSTOM;
    }
}

impl DeviceType {
    /// A device type bitfield with no fields set
    pub const EMPTY: DeviceType = DeviceType(0);
}

bitfield! {
    /// A bitfield representing floating point mode support
    pub struct FPConfig(cl_device_fp_config) {
        pub const DENORM = CL_FP_DENORM;
        pub const INF_NAN = CL_FP_INF_NAN;
        pub const ROUND_TO_NEAREST = CL_FP_ROUND_TO_NEAREST;
        pub const ROUND_TO_ZERO = CL_FP_ROUND_TO_ZERO;
        pub const ROUND_TO_INF = CL_FP_ROUND_TO_INF;
        pub const FMA = CL_FP_FMA;
        pub const CORRECTLY_ROUNDED_DIVIDE_SQRT = CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT;
        pub const SOFT_FLOAT = CL_FP_SOFT_FLOAT;
    }
}

flag_enum! {
    /// Type of cache memory supported by a device
    pub enum MemCacheType(cl_device_mem_cache_type) {
        None = CL_NONE,
        ReadOnly = CL_READ_ONLY_CACHE,
        ReadWrite = CL_READ_WRITE_CACHE,
    }
}

flag_enum! {
    /// Type of local memory for a device
    pub enum LocalMemType(cl_device_local_mem_type) {
        /// No local memory - may only be returned by devices of type CUSTOM
        None = CL_NONE,
        Local = CL_LOCAL,
        Global = CL_GLOBAL,
    }
}

bitfield! {
    pub struct ExecutionCapabilities(cl_device_exec_capabilities) {
        pub const EXEC_KERNEL = CL_EXEC_KERNEL;
        pub const EXEC_NATIVE_KERNEL = CL_EXEC_NATIVE_KERNEL;
    }
}

bitfield! {
    pub struct AffinityDomain(cl_device_affinity_domain) {
        pub const NUMA = CL_DEVICE_AFFINITY_DOMAIN_NUMA;
        pub const L4_CACHE = CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE;
        pub const L3_CACHE = CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE;
        pub const L2_CACHE = CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE;
        pub const L1_CACHE = CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE;
        pub const NEXT_PARTITIONABLE = CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE;
    }
}

bitfield! {
    pub struct SVMCapabilities(cl_device_svm_capabilities) {
        pub const COARSE_GRAIN_BUFFER = CL_DEVICE_SVM_COARSE_GRAIN_BUFFER;
        pub const FINE_GRAIN_BUFFER = CL_DEVICE_SVM_FINE_GRAIN_BUFFER;
        pub const FINE_GRAIN_SYSTEM = CL_DEVICE_SVM_FINE_GRAIN_SYSTEM;
        pub const ATOMICS = CL_DEVICE_SVM_ATOMICS;
    }
}

#[cfg(test)]
mod tests {
    use super::DeviceType;

    #[test]
    fn test_device_type_debug_fmt() {
        assert!(!dbg!(format!("{:?}", DeviceType::EMPTY)).contains("CPU"));
        assert!(dbg!(format!("{:?}", DeviceType::ALL)).contains("CPU"));
        assert!(dbg!(format!("{:?}", DeviceType::CPU | DeviceType::GPU)).contains("GPU"));
    }
}
