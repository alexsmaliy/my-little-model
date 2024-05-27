pub trait TransferFunction {
    fn get_f(&self) -> impl Fn(f32) -> f32;
    fn get_df(&self) -> fn(f32) -> f32;
}

pub struct LeakyReLU {
    slope_lt0: f32,
    slope_gte0: f32,
}

impl TransferFunction for LeakyReLU {
    fn get_f(&self) -> impl Fn(f32) -> f32 {
        move |x| if x < 0f32 { self.slope_lt0 * x } else { self.slope_gte0 * x }
    }
}