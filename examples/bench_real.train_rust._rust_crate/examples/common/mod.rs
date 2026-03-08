#[allow(dead_code)]
pub fn sort_f64_samples(samples: &mut [f64]) {
    samples.sort_by(f64::total_cmp);
}

#[allow(dead_code)]
pub fn median_f64_samples(samples: &mut [f64]) -> f64 {
    sort_f64_samples(samples);
    samples[samples.len() / 2]
}
