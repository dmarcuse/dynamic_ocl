use crate::context::Context;
use crate::program::Program;
use crate::raw::{clBuildProgram, clCreateProgramWithSource, cl_int, cl_program, CL_SUCCESS};
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

pub trait ProgramBuilderType: sealed::ProgramBuilderTypeInternal {}

pub enum FromSource<'a> {
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

pub struct ProgramBuilder<'a, T: ProgramBuilderType> {
    ctx: &'a Context,
    ty: T,
    opts: Option<Cow<'a, str>>,
}

impl<'a> ProgramBuilder<'a, FromSource<'a>> {
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
