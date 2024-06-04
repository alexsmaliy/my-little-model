use std::ops::Neg;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(super) enum Order {
    COLUMNS, ROWS
}

impl Neg for Order {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Order::COLUMNS => Order::ROWS,
            Order::ROWS => Order::COLUMNS,
        }
    }
}
