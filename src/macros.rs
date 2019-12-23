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
             $apiname:ident = $apinamehuman:expr => {
                $(
                    fn $fname:ident ( $( $pname:ident : $pty:ty ),* $(,)? ) $( -> $rty:ty )? ;
                )*
            }
        )*
    ) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub enum OpenCLVersion {
            None,
            $(
                $apiname
            ),*
        }

        impl std::fmt::Display for OpenCLVersion {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                let s = match self {
                    OpenCLVersion::None => "No OpenCL support",
                    $(
                        OpenCLVersion::$apiname => $apinamehuman,
                    )*
                };

                write!(f, "{}", s)
            }
        }

        lazy_static::lazy_static! {
            pub static ref OPENCL_LIB: std::sync::Mutex<Option<Result<OpenCLVersion, &'static dlopen::Error>>> = Default::default();
        }

        static mut OPENCL_VERSION: OpenCLVersion = OpenCLVersion::None;

        mod fnames {
            use const_cstr::const_cstr;
            const_cstr! {
                $(
                    $(
                        pub $fname = stringify!($fname);
                    )*
                )*
            }
        }

        pub unsafe fn load_opencl_internal() -> Result<OpenCLVersion, dlopen::Error> {
            use std::env::var_os;
            use dlopen::utils::platform_file_name;
            use dlopen::raw::Library;
            use dlopen::Error;

            // load library
            // prevent dangling symbols by ensuring it's never dropped
            let name = var_os("OPENCL_LIBRARY").unwrap_or_else(|| platform_file_name("OpenCL"));
            let lib = std::mem::ManuallyDrop::new(Library::open(name)?);

            // set OpenCL version compatibility flags
            $(
                let mut $apiname = true;
            )*

            // load symbols and update version compatibility flags
            $(
                $(
                    let $fname: unsafe extern "C" fn( $( $pname : $pty ),*) $( -> $rty )* = match lib.symbol_cstr(fnames::$fname.as_cstr()) {
                        Ok(addr) => addr,
                        Err(Error::SymbolGettingError(_)) => {
                        $apiname = false;
                            missing_stubs::$fname
                        }
                        Err(e) => return Err(e),
                    };
                )*
            )*

            // set function pointers once all symbols have been loaded
            $(
                $(
                    ptrs::$fname = $fname;
                )*
            )*

            $(
                if $apiname {
                    OPENCL_VERSION = OpenCLVersion::$apiname;
                }
            )*

            Ok(OPENCL_VERSION)
        }

        pub fn load_opencl() -> Result<OpenCLVersion, &'static dlopen::Error> {
            let mut lock = OPENCL_LIB.lock().unwrap();
            if let Some(r) = *lock {
                return r;
            }

            let r = unsafe { load_opencl_internal() }.map_err(|e| Box::leak(Box::new(e)) as &_);
            *lock = Some(r);
            r
        }

        pub mod load_stubs {
            use super::*;

            $(
                $(
                    pub unsafe extern "C" fn $fname( $( $pname : $pty ),* ) $( -> $rty )* {
                        load_opencl().expect("error implicitly loading OpenCL library");
                        ptrs::$fname( $( $pname ),* )
                    }
                )*
            )*
        }

        #[allow(unused_variables)]
        pub mod missing_stubs {
            use super::*;

            $(
                $(
                    pub unsafe extern "C" fn $fname ( $( $pname : $pty ),* ) $( -> $rty )* {
                        panic!("OpenCL library function {} requires {}, but loaded version is {}", stringify!($fname), OpenCLVersion::$apiname, OPENCL_VERSION);
                    }
                )*
            )*
        }

        pub mod ptrs {
            use super::*;

            $(
                $(
                    pub static mut $fname: unsafe extern "C" fn ( $( $pname : $pty ),* ) $( -> $rty )* = load_stubs::$fname;
                )*
            )*
        }
    }
}

macro_rules! wrap_result {
    ( $ctx:expr => $e:expr ) => {
        match $e {
            crate::raw::CL_SUCCESS => Ok(()),
            e => Err(crate::ApiError::new(e, $ctx)),
        }
    };
}

macro_rules! info_funcs {
    (
        $(
            $( #[ $outer:meta ] )*
            pub fn $name:ident(&self) -> $ret:ty = $param:ident;
        )*
    ) => {
        $(
            $( #[ $outer ] )*
            pub fn $name(&self) -> crate::Result<$ret> {
                <Self as crate::util::OclInfo>::get_info(self, crate::raw::$param)
            }
        )*

        fn info_fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.debug_struct(std::any::type_name::<Self>().rsplit("::").next().unwrap())
                $(
                    .field(
                        stringify!($param),
                        match &self.$name() {
                            Ok(v) => v,
                            Err(e) => e,
                        }
                    )
                )*
                .finish()
        }
    }
}

macro_rules! bitfield {
    (
        $( #[ $outer:meta ] )*
        pub struct $name:ident($typ:ty) {
            $(
                $( #[ $inner:meta ] )*
                pub const $vname:ident = $vval:expr;
            )*
        }
    ) => {
        $( #[ $outer ] )*
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        #[repr(transparent)]
        pub struct $name($typ);

        impl $name {
            $(
                $( #[ $inner ] )*
                pub const $vname: $name = $name($vval);
            )*

            /// Create a new wrapped bitfield from the given raw bitfield value.
            ///
            /// # Safety
            ///
            /// This function can be used to create bitfields representing
            /// non-existent flags. You must ensure that the resulting bitfield
            /// is legal anywhere it's used.
            pub const unsafe fn new(value: $typ) -> Self {
                Self(value)
            }

            /// Unwrap this bitfield into the underlying value
            pub const fn raw(self) -> $typ {
                self.0
            }

            /// Check whether this bitfield is equal to or a superset of a given
            /// bitfield
            pub const fn contains(self, other: Self) -> bool {
                self.0 & other.0 == other.0
            }
        }

        impl std::ops::BitOr for $name {
            type Output = Self;

            fn bitor(self, rhs: Self) -> Self {
                Self(self.0 | rhs.0)
            }
        }

        impl std::ops::BitOrAssign for $name {
            fn bitor_assign(&mut self, rhs: Self) {
                self.0 |= rhs.0
            }
        }

        impl std::ops::BitAnd for $name {
            type Output = Self;

            fn bitand(self, rhs: Self) -> Self {
                Self(self.0 & rhs.0)
            }
        }

        impl std::ops::BitAndAssign for $name {
            fn bitand_assign(&mut self, rhs: Self) {
                self.0 &= rhs.0
            }
        }

        impl std::ops::BitXor for $name {
            type Output = Self;

            fn bitxor(self, rhs: Self) -> Self {
                Self (self.0 ^ rhs.0)
            }
        }

        impl std::ops::BitXorAssign for $name {
            fn bitxor_assign(&mut self, rhs: Self) {
                self.0 ^= rhs.0
            }
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                let mut contained: Vec<&'static str> = vec![];

                $(
                    if self.contains(Self::$vname) {
                        contained.push(stringify!($vname));
                    }
                )*

                write!(f, "{}(0b{:b} = {})", stringify!($name), self.0, contained.join(" | "))
            }
        }

        impl crate::util::FromOclInfo for $name {
            fn read<T: crate::util::OclInfo>(from: &T, param_name: T::Param) -> crate::Result<Self> {
                <$typ>::read(from, param_name).map(Self)
            }
        }
    };
}

macro_rules! flag_enum {
    (
        $( #[ $outer:meta ] )*
        pub enum $name:ident($typ:ty) {
            $(
                $( #[ $inner:meta ] )*
                $vname:ident = $vval:path
            ),* $(,)?
        }
    ) => {
        $( #[ $outer ] )*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[repr( u32 )]
        pub enum $name {
            $( $vname = $vval ),*
        }

        impl $name {
            /// Get the raw value of this flag
            pub fn raw(self) -> $typ {
                self as $typ
            }

            pub fn from_raw(value: $typ) -> Option<Self> {
                match value {
                    $(
                        $vval => Some(Self::$vname),
                    )*
                    _ => None
                }
            }
        }

        impl crate::util::FromOclInfo for $name {
            fn read<T: crate::util::OclInfo>(from: &T, param_name: T::Param) -> crate::Result<Self> {
                match <$typ>::read(from, param_name)? {
                    $(
                        $vval => Ok(Self::$vname),
                    )*
                    value => Err(crate::Error::InvalidFlag {
                        value,
                        context: stringify!($name),
                    })
                }
            }
        }
    }
}
