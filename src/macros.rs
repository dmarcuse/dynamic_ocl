/// Create a new opaque type
macro_rules! opaque_type {
    ( $name:ident ) => {
        pub struct $name {
            _opaque: [u8],
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
        pub struct OpenCL {
            $( pub $apiname: $apity, )*
        }
    }
}
