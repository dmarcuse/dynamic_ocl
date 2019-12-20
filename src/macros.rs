/// Create a new opaque type
macro_rules! opaque_type {
    ( $name:ident ) => {
        #[doc(hidden)]
        pub struct $name {
            _opaque: (),
        }
    };

    ( $( $name:ident),* $(,)? ) => {
        $( opaque_type!{$name} )*
    };
}

/// Define OpenCL error code constants and a function to get the name of an
/// error code
macro_rules! error_codes {
    ( $($name:ident = $value:expr),* $(,)? ) => {
        $( pub const $name: i32 = $value; )*

        /// Get the name of an OpenCL error code, returning `None` if the error
        /// code is unknown
        pub fn error_name(code: i32) -> Option<&'static str> {
            match code {
                $($name => Some(stringify!($name)),)*
                _ => None,
            }
        }
    };
}

/// Define raw OpenCL function bindings
macro_rules! raw_functions {
    (
        $(
             $apiname:ident : $apity:ty {
                $(
                    fn $fname:ident ( $( $pname:ident : $pty:ty ),* $(,)? ) $( -> $rty:ty )? ;
                )*
            }
        )*
    ) => {
        use dlopen_derive::{WrapperApi, WrapperMultiApi};
        use dlopen::wrapper::{WrapperApi, WrapperMultiApi};

        $(
            #[derive(WrapperApi)]
            pub struct $apiname {
                $(
                    $fname: unsafe extern "C" fn ( $( $pname : $pty ),* ) $( -> $rty )*
                ),*
            }

            impl std::fmt::Debug for $apiname {
                fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    f.debug_struct(stringify!($apiname))
                        $( .field(stringify!($fname), &(self.$fname as *const ())) )*
                        .finish()
                }
            }
        )*

        #[derive(Debug, WrapperMultiApi)]
        pub struct RawOpenCL {
            $( pub $apiname: $apity, )*
        }
    }
}

macro_rules! ocl_try {
    ( $ctx:expr => $e:expr ) => {
        match $e {
            crate::raw::CL_SUCCESS => {}
            e => return Err(crate::ApiError::new(e, $ctx).into()),
        }
    };
}

macro_rules! info_func_ret_type {
    ( get_info_string ) => { crate::Result<std::ffi::CString> }
}

macro_rules! info_funcs {
    (
        $(
            $( #[ $outer:meta ] )*
            pub fn $name:ident(&self) => self.$delegate:ident($param:ident);
        )*
    ) => {
        $(
            $( #[ $outer ] )*
            pub fn $name(&self) -> info_func_ret_type!($delegate) {
                <Self as crate::util::OclInfo>::$delegate(self, crate::raw::$param)
            }
        )*
    }
}
