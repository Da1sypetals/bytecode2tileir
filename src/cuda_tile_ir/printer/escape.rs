pub fn escape_mlir_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\n', "\\0A")
        .replace('\0', "\\00")
        .replace('"', "\\22")
}
