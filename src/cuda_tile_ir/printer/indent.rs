use std::fmt::{self, Write};

pub trait MlirPrinter: Write {
    fn write_indent(&mut self, level: usize) -> fmt::Result {
        for _ in 0..level {
            self.write_char(' ')?;
        }
        Ok(())
    }

    fn writeln(&mut self, s: &str) -> fmt::Result {
        self.write_str(s)?;
        self.write_char('\n')
    }
}

impl MlirPrinter for String {}
impl<W: Write + ?Sized> MlirPrinter for &mut W {}
