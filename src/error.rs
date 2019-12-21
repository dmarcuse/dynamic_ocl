use crate::raw::{cl_int, cl_uint, error_name};
use std::convert::Infallible;
use std::fmt::{self, Debug, Display, Formatter};

/// An error code returned by an OpenCL API call
#[derive(thiserror::Error)]
pub struct ApiError {
    code: cl_int,
    context: &'static str,
}

impl ApiError {
    /// Create a new `ApiError` with the given error code and context
    pub fn new(code: cl_int, context: &'static str) -> Self {
        Self { code, context }
    }
}

impl Debug for ApiError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        <Self as Display>::fmt(self, f)
    }
}

impl Display for ApiError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "{}: OpenCL error {} ({})",
            self.context,
            self.code,
            error_name(self.code).unwrap_or("unknown error code")
        )
    }
}

#[derive(thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    ApiError(#[from] ApiError),

    #[error("Invalid flag value {value:x} for type {context}")]
    InvalidFlag {
        value: cl_uint,
        context: &'static str,
    },
}

impl Debug for Error {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        <Self as Display>::fmt(self, f)
    }
}

impl From<Infallible> for Error {
    fn from(_: Infallible) -> Self {
        panic!("impossible condition");
    }
}

/// An OpenCL result type
pub type Result<T> = std::result::Result<T, Error>;
