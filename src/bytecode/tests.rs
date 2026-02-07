#[cfg(test)]
mod tests {
    use crate::bytecode::*;

    #[test]
    fn test_magic_detection() {
        let valid = [0x7F, b'T', b'i', b'l', b'e', b'I', b'R', 0x00];
        assert!(is_tilir_bytecode(&valid));

        let invalid = [0x7F, b'E', b'L', b'F', 0x00, 0x00, 0x00, 0x00];
        assert!(!is_tilir_bytecode(&invalid));
    }

    #[test]
    fn test_varint_encoding() {
        use crate::bytecode::reader::{ByteRead, Cursor};

        let data = [0x05];
        let mut r = Cursor::new(&data);
        assert_eq!(r.read_var_u64().unwrap(), 5);

        let data = [0x80, 0x01];
        let mut r = Cursor::new(&data);
        assert_eq!(r.read_var_u64().unwrap(), 128);

        let data = [0x01];
        let mut r = Cursor::new(&data);
        assert_eq!(r.read_var_i64().unwrap(), -1);
    }

    #[test]
    fn test_type_tags() {
        use crate::bytecode::tags::TypeTag;

        assert_eq!(TypeTag::try_from(0).unwrap(), TypeTag::I1);
        assert_eq!(TypeTag::try_from(7).unwrap(), TypeTag::F32);
        assert_eq!(TypeTag::try_from(13).unwrap(), TypeTag::Tile);
        assert!(TypeTag::try_from(99).is_err());
    }

    #[test]
    fn test_reader_operations() {
        use crate::bytecode::reader::{ByteRead, Cursor};

        let data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let mut r = Cursor::new(&data);

        assert_eq!(r.read_u8().unwrap(), 0x01);
        assert_eq!(r.remaining(), 7);

        assert_eq!(r.read_u16_le().unwrap(), 0x0302);
        assert_eq!(r.remaining(), 5);

        let _ = r.read_bytes(2).unwrap();
        assert_eq!(r.remaining(), 3);
    }

    #[test]
    fn test_padding_alignment() {
        use crate::bytecode::reader::{ByteRead, Cursor};

        let data = [0xCB; 16];
        let mut r = Cursor::new(&data);

        let _ = r.read_bytes(1).unwrap();
        assert_eq!(r.pos(), 1);

        r.align_to(4, 0xCB).unwrap();
        assert_eq!(r.pos(), 4);

        let _ = r.read_bytes(1).unwrap();
        r.align_to(8, 0xCB).unwrap();
        assert_eq!(r.pos(), 8);
    }

    #[test]
    fn test_invalid_bytecode() {
        let invalid_data = [0x00, 0x01, 0x02, 0x03];
        let result = BytecodeFile::parse(&invalid_data);
        assert!(result.is_err());
    }
}
