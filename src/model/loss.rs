use crate::linalg::Vector;

type V<const N: usize> = Vector<N>;

pub trait LossFunction: Copy {
    fn get_L<const DIM: usize>(&self) -> impl Fn(&V<DIM>, &V<DIM>) -> (f32, V<DIM>) + 'static;
    fn get_dL_da<const DIM: usize>(&self) -> impl Fn(&V<DIM>, &V<DIM>) -> V<DIM> + 'static;
}

#[derive(Clone, Copy)]
pub struct MeanSquaredErrorLoss;

impl LossFunction for MeanSquaredErrorLoss {
    fn get_L<const DIM: usize>(&self) -> impl Fn(&V<DIM>, &V<DIM>) -> (f32, V<DIM>) + 'static {
        |target: &V<DIM>, output: &V<DIM>| {
            let errors = target - output;
            (errors.sum_of_squares(), errors)
        }
    }

    fn get_dL_da<const DIM: usize>(&self) -> impl Fn(&V<DIM>, &V<DIM>) -> V<DIM> + 'static {
        |target: &V<DIM>, output: &V<DIM>| -2f32 * &(target - output)
    }
}

#[derive(Clone, Copy)]
pub struct SoftmaxCrossEntropyLoss;

impl LossFunction for SoftmaxCrossEntropyLoss {
    fn get_L<const DIM: usize>(&self) -> impl Fn(&V<DIM>, &V<DIM>) -> (f32, V<DIM>) + 'static {
        |target: &V<DIM>, output: &V<DIM>| {
            let sum_exp = output.map(<f32>::exp).sum();
            let mut entropy = 0f32;
            for i in 0..DIM {
                // Note minus: cross-entropy is -sum over i of { p(i) * log(q(i)) }.
                entropy -= target[i] * (output[i].exp() / sum_exp).ln();
            }
            (entropy, V::zero()) // TODO: remove dummy errors.
        }
    }

    /**
        $$
            \begin{array}{rrll}
            & H(p, q) & = & -\sum_{i}{p_{i} \ln{q_{i}}} \\
            & \mathrm{softmax}(q) & = & \left[ \ldots, \exp{q_i} / \sum_j{\exp q_j}, \ldots \right] \\
            \text{Take:} & p & = & \Bigl[p_0, p_1, p_2\Bigr] \\
            & q & = & \Bigl[q_0, q_1, q_2\Bigr] \\
            \text{Then:} & & & \\
            & \dfrac{\partial{H(p, \mathrm{softmax}(q))}}{\partial{q_0}} & = & -\dfrac{\partial}{\partial{q_o}} \Biggl[ p_1 \ln\dfrac{e^{q_0}}{(e^{q_o} + e^{q_1} + e^{q_2})} + p_1 \ln\dfrac{e^{q_1}}{(e^{q_0} + e^{q_1} + e^{q_2})} + p_2 \ln\dfrac{e^{q_2}}{(e^{q_0} + e^{q_1} + e^{q_2})} \Biggr] \\
            & & = & -\Biggl[ p_0\left(\dfrac{e^{q_0} + e^{q_1} + e^{q_2}}{ e^{q_0}}\right)\left(\dfrac{e^{a_0}\left(e^{q_0} + e^{q_1} + e^{q_2}\right) - (e^{a_0})^2}{(e^{q_0} + e^{q_1} + e^{q_2})^2}\right) + p_1\left(\dfrac{e^{q_0} + e^{q_1} + e^{q_2}}{e^{q_1}}\right)\left(-\dfrac{e^{q_o}e^{q_1}}{(e^{q_0} + e^{q_1} + e^{q_2})^2}\right) + p_2\left(\dfrac{e^{q_0} + e^{q_1} + e^{q_2}}{e^{q_2}}\right)\left(-\dfrac{e^{q_0}e^{q_2}}{(e^{q_0} + e^{q_1} + e^{q_2})^2}\right) \Biggr] \\
            & & = & -\Biggl[ p_0\left(\cancel{\dfrac{e^{q_0} + e^{q_1} + e^{q_2}}{ e^{q_0}}}\right)\left(\dfrac{\cancel{e^{a_0}}\left(e^{q_0} + e^{q_1} + e^{q_2}\right) - (e^{a_0})^\cancel{2}}{(e^{q_0} + e^{q_1} + e^{q_2})^\cancel{2}}\right) + p_1\left(\cancel{\dfrac{e^{q_0} + e^{q_1} + e^{q_2}}{e^{q_1}}}\right)\left(-\dfrac{e^{q_0}\cancel{e^{q_1}}}{(e^{q_0} + e^{q_1} + e^{q_2})^\cancel{2}}\right) + p_2\left(\cancel{\dfrac{e^{q_0} + e^{q_1} + e^{q_2}}{e^{q_2}}}\right)\left(-\dfrac{e^{q_0}\cancel{e^{q_2}}}{(e^{q_0} + e^{q_1} + e^{q_2})^\cancel{2}}\right) \Biggr] \\
            & & = & -\Biggl[ p_0 \dfrac{e^{q_1} + e^{q_2}}{e^{q_0} + e^{q_1} + e^{q_2}} - p_1 \dfrac{e^{q_0}}{e^{q_0} + e^{q_1} + e^{q_2}} - p_2 \dfrac{e^{q_0}}{e^{q_0} + e^{q_1} + e^{q_2}}\Biggr] \\
            & & = & -\Biggl[ \dfrac{p_0 e^{q_1} + p_0 e^{q_2} - p_1 e^{q_0} - p_2 e^{q_0}}{e^{q_0} + e^{q_1} + e^{q_2}} \Biggr] \\
            & & = & -\Biggl[ \dfrac{p_0 e^{q_1} + p_0 e^{q_2} + (p_0 e^{q_0} - p_0 e^{q_0}) - p_1 e^{q_0} - p_2 e^{q_0}}{e^{q_0} + e^{q_1} + e^{q_2}} \Biggr] \\
            & & = & -\Biggl[ \dfrac{p_0(e^{q_0} + e^{q_1} + e^{q_2}) - e^{q_0}(p_0+p_1+p_2)}{e^{q_0} + e^{q_1} + e^{q_2}} \Biggr] \\
            & & = & -\Biggl[ p_0 - \dfrac{e^{q_0}(p_0+p_1+p_2)}{e^{q_0} + e^{q_1} + e^{q_2}} \Biggr] \\
            & & = & \Biggl(\dfrac{\sum_j{}p_j}{\sum_j{e^{q_j}}}\Biggr)e^{q_0} - p_0 \\
            \text{Waving hands vigorously:} & & & \\
            & \dfrac{\partial{H(p, \mathrm{softmax}(q))}}{\partial{q_i}} & = & \Biggl(\dfrac{\sum_j{}p_j}{\sum_j{e^{q_j}}}\Biggr)e^{q_i} - p_i \\
            \text{If }\sum_j{p_j}=1\text{, this reduces to: } & & & \\
            & \dfrac{\partial{H(p, \mathrm{softmax}(q))}}{\partial{q}} & = & \mathrm{softmax}(q) - p.\quad\blacksquare
            \end{array}
        $$
    */
    fn get_dL_da<const DIM: usize>(&self) -> impl Fn(&V<DIM>, &V<DIM>) -> V<DIM> + 'static {
        |target: &V<DIM>, output: &V<DIM>| {
            let sum_pj = target.sum(); // Normally sum(p) = 1;
            let sum_exp_qj = output.map(<f32>::exp).sum();
            let ratio = sum_pj / sum_exp_qj; // If sum(p) = 1, this is just the softmax denominator.
            let mut dL_da = V::zero();
            for i in 0..DIM {
                // If sum(p) = 1, this is just softmax(q) - target.
                dL_da[i] = output[i].exp() * ratio - target[i];
            }
            dL_da
        }
    }
}
