#[cfg(test)]
pub fn init_test_logger() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
        .format_timestamp_secs()
        .format_target(false)
        .is_test(true)
        .try_init();
}
