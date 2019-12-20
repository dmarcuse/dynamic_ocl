use crate::raw::{cl_int, error_name};
use std::fmt::{self, Debug, Display, Formatter};

/// An error code returned by an OpenCL API call
#[derive(Debug, thiserror::Error)]
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

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    ApiError(#[from] ApiError),
}

/// An OpenCL result type
pub type Result<T> = std::result::Result<T, Error>;
