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

macro_rules! info_funcs {
    (
        $(
            $( #[ $outer:meta ] )*
            pub fn $name:ident(&self) -> $ret:ty = self.$delegate:ident($param:ident);
        )*
    ) => {
        $(
            $( #[ $outer ] )*
            pub fn $name(&self) -> crate::Result<$ret> {
                <Self as crate::util::OclInfo>::$delegate(self, crate::raw::$param)
                    .and_then(|v| crate::util::info_convert(&v))
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

        impl crate::util::OclInfoFrom<$typ> for $name {
            fn convert(&value: &$typ) -> crate::Result<Self> {
                Ok(Self(value))
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

        impl crate::util::OclInfoFrom<$typ> for $name {
            fn convert(&value: &$typ) -> crate::Result<Self> {
                Self::from_raw(value)
                    .ok_or_else(|| crate::Error::InvalidFlag { value, context: stringify!($name) })
            }
        }
    }
}
