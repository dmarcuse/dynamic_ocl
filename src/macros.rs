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

        /// OpenCL version supported by this system - only set once OpenCL lib
        /// has been loaded.
        pub static mut SYSTEM_OPENCL_VERSION: OpenCLVersion = OpenCLVersion::None;

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
                    SYSTEM_OPENCL_VERSION = OpenCLVersion::$apiname;
                }
            )*

            Ok(SYSTEM_OPENCL_VERSION)
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

        mod load_stubs {
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
        mod missing_stubs {
            use super::*;

            $(
                $(
                    pub unsafe extern "C" fn $fname ( $( $pname : $pty ),* ) $( -> $rty )* {
                        panic!("OpenCL library function {} requires {}, but loaded version is {}", stringify!($fname), OpenCLVersion::$apiname, SYSTEM_OPENCL_VERSION);
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

/// Wrap an OpenCL error code in a result
#[cfg(feature = "safe")]
macro_rules! wrap_result {
    ( $ctx:expr => $e:expr ) => {
        match $e {
            crate::raw::CL_SUCCESS => Ok(()),
            e => Err(crate::ApiError::new(e, $ctx)),
        }
    };
}

macro_rules! check_ocl_version {
    ( $context:expr => $version:ident ) => {
        if crate::raw::SYSTEM_OPENCL_VERSION <= crate::raw::OpenCLVersion::$version {
            Err(crate::Error::UnsupportedVersion {
                expected: crate::raw::OpenCLVersion::$version,
                actual: crate::raw::SYSTEM_OPENCL_VERSION,
                context: $context,
            })
        } else {
            Ok(())
        }
    };
}

/// Define OpenCL info functions
#[cfg(feature = "safe")]
macro_rules! info_funcs {
    (
        $(
            $( #[ $outer:meta ] )*
            $vis:vis fn $name:ident(&self) -> $ret:ty = $param:ident;
        )*
    ) => {
        $(
            $( #[ $outer ] )*
            $vis fn $name(&self) -> crate::Result<$ret> {
                <Self as crate::util::OclInfo>::get_info(self, crate::raw::$param)
            }
        )*

        #[allow(dead_code)]
        fn info_fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.debug_struct(&tynm::type_name::<Self>())
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

/// Define a wrapper for an OpenCL bitfield type
#[cfg(feature = "safe")]
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

/// Define a wrapper for an OpenCL flag type, with a fixed set of valid values
#[cfg(feature = "safe")]
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
            $( $vname = $vval as u32),*
        }

        impl $name {
            /// Get the raw value of this flag
            pub const fn raw(self) -> $typ {
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
                match u32::read(from, param_name)? as $typ {
                    $(
                        $vval => Ok(Self::$vname),
                    )*
                    value => Err(crate::Error::InvalidFlag {
                        value: value as u32,
                        context: stringify!($name),
                    })
                }
            }
        }
    }
}

/// Count the length of a tuple type
#[cfg(feature = "safe")]
macro_rules! tuple_len {
    ( () ) => { 0 };
    ( ( $head:ty $( , $tail:ty )* $(,)? ) ) => { 1 + tuple_len! { ( $( $tail ),* ) } };
}

/// Define KernelArgList implementations for tuple types
#[cfg(feature = "safe")]
macro_rules! kernel_arg_list_tuples {
    (
        $(
            $( #[ $meta:meta ] )*
            ( $( $tyvar:ident ),* $(,)? )
        ),* $(,)?
    ) => {
        $(
            #[allow(unused_parens)]
            impl<
                $( $tyvar : KernelArg ),*
            > sealed::KernelArgListInternal for (
                $( $tyvar ),*
            ) {
                #[allow(unused_variables, unused_assignments, unused_mut, non_snake_case)]
                fn bind(self, kernel: UnboundKernel, type_checks: bool) -> Result<Kernel<Self>> {
                    let mut idx = 0;
                    let ( $( $tyvar ),* ) = self;

                    $(
                        let $tyvar = Bound::bind(&kernel, idx, $tyvar, type_checks)?;
                        idx += 1;
                    )*

                    Ok(Kernel {
                        kernel,
                        args: ( $( $tyvar ),* )
                    })
                }
            }

            $( #[ $meta ] )*
            #[allow(unused_parens)]
            impl<
                $( $tyvar : KernelArg ),*
            > KernelArgList for (
                $( $tyvar ),*
            ) {
                type Bound = (
                    $( Bound< $tyvar > ),*
                );

                const NUM_ARGS: usize = tuple_len! { ( $( $tyvar ),* ) };
            }

            #[allow(unused_parens)]
            impl<
                'a,
                $( $tyvar : 'a + KernelArg ),*
            > BindProject<'a> for ( $( Bound<$tyvar> ),* ) {
                type Projected = (
                    $( Pin<&'a mut Bound<$tyvar>> ),*
                );
            }

            #[allow(unused_parens)]
            impl<
                'a,
                $( $tyvar : 'a + KernelArg ),*
            > sealed::BindProjectInternal<'a> for ( $( Bound<$tyvar> ),* ) {
                #[allow(non_snake_case)]
                fn project(self: Pin<&'a mut Self>) -> <Self as BindProject<'a>>::Projected {
                    unsafe {
                        let ( $( $tyvar ),* ) = self.get_unchecked_mut();
                        ( $( Pin::new_unchecked($tyvar) ),* )
                    }
                }
            }
        )*
    }
}
