use std::ops::Neg;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
/// Indicates whether strides of an array underlying a dense matrix
/// should be interpreted as matrix rows or matrix columns. Twiddling
/// this parameter is a simple indicator of transposing a matrix.
pub(super) enum Order {
    COLS,
    ROWS,
}

impl Neg for Order {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Order::COLS => Order::ROWS,
            Order::ROWS => Order::COLS,
        }
    }
}
