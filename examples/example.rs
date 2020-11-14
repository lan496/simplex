use simplex::{StandardForm, LPResult, simplex};
use approx::{assert_relative_eq};

fn main() {
    // maximize   x0 - x1 + x2
    // subject to 2 x0 -   x1 + 2 x2 <= 4
    //            2 x0 - 3 x1 +   x2 <= -5
    //             -x0 +   x1 - 2 x2 <= -1
    let c = vec![1.0, -1.0, 1.0];
    let b = vec![4.0, -5.0, -1.0];
    let a = vec![
        vec![2.0, -1.0, 2.0],
        vec![2.0, -3.0, 1.0],
        vec![-1.0, 1.0, -2.0],
    ];

    let standard = StandardForm::new(c, a, b);
    let solution = simplex(&standard);
    match solution {
        LPResult::Feasible((sol, optimal)) => {
            assert_relative_eq!(sol[..], [0.0, 2.8, 3.4]);
            assert_relative_eq!(optimal, 0.6, epsilon = 1e-8);
            println!("solution: {:?}", sol);
            println!("optimal: {}", optimal);
        },
        LPResult::Infeasible => {
            println!("Infeasible");
        },
        LPResult::Unbounded => {
            println!("Unbounded");
        },
    };

}
