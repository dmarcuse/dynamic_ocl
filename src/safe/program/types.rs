use crate::context::Context;
use crate::program::Program;
use crate::raw::{
    clBuildProgram, clCreateProgramWithSource, cl_build_status, cl_int, cl_program,
    cl_program_binary_type, CL_BUILD_ERROR, CL_BUILD_IN_PROGRESS, CL_BUILD_NONE, CL_BUILD_SUCCESS,
    CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT, CL_PROGRAM_BINARY_TYPE_EXECUTABLE,
    CL_PROGRAM_BINARY_TYPE_LIBRARY, CL_PROGRAM_BINARY_TYPE_NONE, CL_SUCCESS,
};
use crate::Result;
use sealed::ProgramBuilderTypeInternal;
use std::borrow::Cow;
use std::ffi::CString;
use std::ptr::null_mut;

mod sealed {
    use super::ProgramBuilder;
    use crate::program::ProgramBuilderType;
    use crate::raw::{cl_int, cl_program};

    pub trait ProgramBuilderTypeInternal {
        const CONTEXT: &'static str;

        unsafe fn create_program(builder: &ProgramBuilder<Self>, err: *mut cl_int) -> cl_program
        where
            Self: Sized + ProgramBuilderType;
    }
}

/// A program builder type, specifying how a program should be built (e.g.
/// compiled from source code, loaded from a binary, etc)
pub trait ProgramBuilderType: sealed::ProgramBuilderTypeInternal {}

/// A `ProgramBuilderType` implementation for programs to be compiled from
/// source
pub enum FromSource<'a> {
    /// Build the program from a single source file
    Single(&'a [u8]),
}

impl<'a> ProgramBuilderTypeInternal for FromSource<'a> {
    const CONTEXT: &'static str = "clCreateProgramWithSource";

    unsafe fn create_program(builder: &ProgramBuilder<Self>, err: *mut cl_int) -> cl_program {
        match builder.ty {
            FromSource::Single(src) => clCreateProgramWithSource(
                builder.ctx.raw(),
                1,
                [src.as_ptr()].as_ptr() as _,
                [src.len()].as_ptr(),
                err,
            ),
        }
    }
}

impl<'a> ProgramBuilderType for FromSource<'a> {}

/// A partially built OpenCL program
#[must_use]
pub struct ProgramBuilder<'a, T: ProgramBuilderType> {
    ctx: &'a Context,
    ty: T,
    opts: Option<Cow<'a, str>>,
}

impl<'a> ProgramBuilder<'a, FromSource<'a>> {
    /// Begin building a program with a single source file
    pub fn with_source(ctx: &'a Context, src: &'a impl AsRef<[u8]>) -> Self {
        Self {
            ctx,
            ty: FromSource::Single(src.as_ref()),
            opts: None,
        }
    }
}

impl<'a, T: ProgramBuilderType> ProgramBuilder<'a, T> {
    /// Append an option to be passed to the compiler
    pub fn opt(&mut self, opts: impl Into<Cow<'a, str>>) -> &mut Self {
        match &mut self.opts {
            Some(old) => {
                *old += " ";
                *old += opts.into();
            }
            o => *o = Some(opts.into()),
        };
        self
    }

    /// Build the program
    pub fn build(&self) -> Result<Program> {
        unsafe {
            let mut err = CL_SUCCESS;

            let program = T::create_program(self, &mut err as _);
            wrap_result!(T::CONTEXT => std::mem::replace(&mut err, CL_SUCCESS))?;

            let opts = self
                .opts
                .as_ref()
                .map(|o| CString::new(o.as_bytes()).unwrap());

            wrap_result!("clBuildProgram" => clBuildProgram(
                program,
                0,
                null_mut(),
                opts.map(|o| o.as_ptr()).unwrap_or(null_mut()),
                None,
                null_mut()
            ))?;

            Ok(Program(program))
        }
    }
}

flag_enum! {
    pub enum ProgramBuildStatus(cl_build_status) {
        None = CL_BUILD_NONE,
        InProgress = CL_BUILD_IN_PROGRESS,
        Success = CL_BUILD_SUCCESS,
        Error = CL_BUILD_ERROR,
    }
}

flag_enum! {
    pub enum ProgramBinaryType(cl_program_binary_type) {
        None = CL_PROGRAM_BINARY_TYPE_NONE,
        CompiledObject = CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT,
        Library = CL_PROGRAM_BINARY_TYPE_LIBRARY,
        Executable = CL_PROGRAM_BINARY_TYPE_EXECUTABLE
    }
}
