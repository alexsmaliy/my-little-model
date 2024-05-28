pub trait ActivationFunction {
    fn get_f(&self) -> impl Fn(f32) -> f32 + 'static;
    fn get_df(&self) -> impl Fn(f32) -> f32 + 'static;
}

pub struct Identity {}

impl ActivationFunction for Identity {
    fn get_f(&self) -> impl Fn(f32) -> f32 + 'static {
        |x| x
    }

    fn get_df(&self) -> impl Fn(f32) -> f32 + 'static {
        |_| 1f32
    }
}

pub struct LeakyReLU {
    pub slope_lt0: f32,
    pub slope_gte0: f32,
}

impl ActivationFunction for LeakyReLU {
    fn get_f(&self) -> impl Fn(f32) -> f32 + 'static {
        let (lt0, gte0) = (self.slope_lt0, self.slope_gte0);
        move |x| if x < 0f32 { lt0 * x } else { gte0 * x }
    }

    fn get_df(&self) -> impl Fn(f32) -> f32 + 'static {
        let (lt0, gte0) = (self.slope_lt0, self.slope_gte0);
        move |x| if x < 0f32 { lt0 } else { gte0 }
    }
}

pub struct ReLU {
    slope_gte0: f32,
}

impl ActivationFunction for ReLU {
    fn get_f(&self) -> impl Fn(f32) -> f32 + 'static {
        let gte0 = self.slope_gte0;
        move |x| if x < 0f32 { 0f32 } else { gte0 * x }
    }

    fn get_df(&self) -> impl Fn(f32) -> f32 + 'static {
        let gte0 = self.slope_gte0;
        move |x| if x < 0f32 { 0f32 } else { gte0 }
    }
}
