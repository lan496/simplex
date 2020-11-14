use std::collections::HashSet;

const EPS: f64 = 1e-8;

/// Standard form for LP
///
/// maximize   sum_{j} c[j] * x[j]
/// subject to sum_{j} a[i][j] * x[j] <= b[j] (for all i)
///            x[j] >= 0 (for all j)
#[derive(Debug, Clone)]
pub struct StandardForm {
    c: Vec<f64>,
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
}

impl StandardForm {
    pub fn new(c: Vec<f64>, a: Vec<Vec<f64>>, b: Vec<f64>) -> Self {
        let dimensions = c.len();
        let num_constraints = b.len();

        if a.len() != num_constraints {
            panic!();
        }
        for ai in a.iter() {
            if ai.len() != dimensions {
                panic!();
            }
        }

        StandardForm { c, a, b }
    }

    fn into_slack_from(&self) -> SlackForm {
        let num_nonbasic = self.c.len();
        let num_basic = self.b.len();
        let num_variables = num_nonbasic + num_basic;

        let mut nonbasic = HashSet::new();
        for j in 0..num_nonbasic {
            nonbasic.insert(j);
        }

        let mut basic = HashSet::new();
        for i in num_nonbasic..num_variables {
            basic.insert(i);
        }

        let mut a = vec![vec![0.0; num_variables]; num_variables];
        for &i in basic.iter() {
            for &j in nonbasic.iter() {
                a[i][j] = self.a[i - num_basic][j];
            }
        }

        let mut b = vec![0.0; num_variables];
        for &i in basic.iter() {
            b[i] = self.b[i - num_basic];
        }

        let mut c = vec![0.0; num_variables];
        for &j in nonbasic.iter() {
            c[j] = self.c[j];
        }

        SlackForm {
            nonbasic,
            basic,
            a,
            b,
            c,
            v: 0.0,
        }
    }

    /// return slack form of auxiliary problem of the original standard form
    /// variables are ordered by [nonbasic, basic, auxiliary_variable]
    fn into_auxiliary_slack_form(&self) -> SlackForm {
        let num_nonbasic = self.c.len();
        let num_basic = self.b.len();
        let num_variables = num_nonbasic + num_basic;

        let mut nonbasic = HashSet::new();
        for j in 0..num_nonbasic {
            nonbasic.insert(j);
        }

        let mut basic = HashSet::new();
        for i in num_nonbasic..num_variables {
            basic.insert(i);
        }

        let mut a = vec![vec![0.0; num_variables + 1]; num_variables + 1];
        for &i in basic.iter() {
            for &j in nonbasic.iter() {
                a[i][j] = self.a[i - num_nonbasic][j];
            }
        }

        let mut b = vec![0.0; num_variables + 1];
        for &i in basic.iter() {
            b[i] = self.b[i - num_nonbasic];
        }

        let vaux = num_variables; // new variable for auxiliary problem
        nonbasic.insert(vaux);
        for &i in basic.iter() {
            a[i][vaux] = -1.0;
        }
        let mut c = vec![0.0; num_variables + 1];
        c[vaux] = -1.0;

        SlackForm {
            nonbasic,
            basic,
            a,
            b,
            c,
            v: 0.0,
        }
    }
}

/// Slack form for LP
///
/// z = v + sum_{j in nonbasic} c[j] * x[j]
/// x[i] = b[i] - sum_{j in nonbasic} a[i][j] * x[j] (for i in basic)
#[derive(Debug, Clone)]
pub struct SlackForm {
    nonbasic: HashSet<usize>, // N
    basic: HashSet<usize>,    // B
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
    c: Vec<f64>,
    v: f64,
}

impl SlackForm {
    /// pivot basic variable `leaving` and nonbasic variable `entering`
    fn pivot(&self, leaving: usize, entering: usize) -> Self {
        let num_nonbasic = self.nonbasic.len();
        let num_basic = self.basic.len();
        let num_variables = num_nonbasic + num_basic;

        let mut nonbasic= self.nonbasic.clone();
        nonbasic.remove(&entering);

        let mut basic = self.basic.clone();
        basic.remove(&leaving);

        let mut b = vec![0.0; num_variables];
        b[entering] = self.b[leaving] / self.a[leaving][entering];
        for &i in self.basic.iter() {
            b[i] = self.b[i] - self.a[i][entering] * b[entering];
        }

        let mut a = vec![vec![0.0; num_variables]; num_variables];
        a[entering][leaving] = 1.0 / self.a[leaving][entering];
        for &j in self.nonbasic.iter() {
            a[entering][j] = self.a[leaving][j] / self.a[leaving][entering];
        }
        for &i in self.basic.iter() {
            for &j in self.nonbasic.iter() {
                a[i][j] = self.a[i][j] - self.a[i][entering] * a[entering][j];
            }
            a[i][leaving] = -self.a[i][entering] * a[entering][leaving]
        }

        let mut c = vec![0.0; num_variables];
        c[leaving] = -self.c[entering] * a[entering][leaving];
        for &j in self.nonbasic.iter() {
            c[j] = self.c[j] - self.c[entering] * a[entering][j];
        }

        let v = self.v + self.c[entering] * b[entering];

        nonbasic.insert(leaving);
        basic.insert(entering);

        SlackForm {
            nonbasic,
            basic,
            a,
            b,
            c,
            v,
        }
    }

    fn get_basic_solution(&self) -> Vec<f64> {
        let mut solution = vec![0.0; self.a.len()];
        for &i in self.basic.iter() {
            solution[i] = self.b[i];
        }
        solution
    }

    fn get_objective(&self, solution: &Vec<f64>) -> f64 {
        assert!(solution.len() == self.c.len());
        let mut obj = self.v;
        for &j in self.nonbasic.iter() {
            obj += self.c[j] * solution[j];
        }
        obj
    }
}

#[derive(Debug)]
pub enum LPResult {
    Feasible((Vec<f64>, f64)),
    Infeasible,
    Unbounded,
}

fn initialize_simplex(standard: &StandardForm) -> Option<SlackForm> {
    let mut k = 0;
    let mut bmin = standard.b[k];
    for i in 0..standard.b.len() {
        if bmin - standard.b[i] > EPS {
            k = i;
            bmin = standard.b[i];
        }
    }
    if bmin >= -EPS {
        return Some(standard.into_slack_from());
    }

    let num_nonbasic = standard.c.len();
    let num_basic = standard.b.len();
    let num_variables = num_nonbasic + num_basic;

    let mut slack_aux = standard.into_auxiliary_slack_form();
    let vaux = num_variables;

    // pivot `k` and auxiliary variable
    slack_aux = slack_aux.pivot(num_nonbasic + k, vaux);

    while let Some(entering) = slack_aux.c.iter().position(|&ci| ci > EPS) {
        let mut leaving = 0;
        let mut delta = f64::INFINITY;
        for &ii in slack_aux.basic.iter().filter(|&i| slack_aux.a[*i][entering] > EPS) {
            let dii = slack_aux.b[ii] / slack_aux.a[ii][entering];
            if delta - dii > EPS {
                delta = dii;
                leaving = ii;
            }
        }
        slack_aux = slack_aux.pivot(leaving, entering);
    }

    let basic_solution = slack_aux.get_basic_solution();
    if basic_solution[num_variables].abs() <= EPS {
        if slack_aux.basic.contains(&vaux) {
            let entering = slack_aux.nonbasic.iter().find(|&i| slack_aux.a[num_variables][*i].abs() > EPS).unwrap();
            slack_aux = slack_aux.pivot(vaux, *entering);
        }

        let mut nonbasic = slack_aux.nonbasic.clone();
        nonbasic.remove(&vaux);
        let basic = slack_aux.basic.clone();

        let mut a = vec![vec![0.0; num_variables]; num_variables];
        for i in 0..num_variables {
            for j in 0..num_variables {
                a[i][j] = slack_aux.a[i][j];
            }
        }
        let mut b = vec![0.0; num_variables];
        for i in 0..num_variables {
            b[i] = slack_aux.b[i];
        }

        let mut v = 0.0;
        for &i in slack_aux.basic.iter() {
            if i < num_nonbasic {
                v += standard.c[i] * slack_aux.b[i];
            }
        }
        let mut c = vec![0.0; num_variables];
        for &j in nonbasic.iter() {
            if j < num_nonbasic {
                c[j] += standard.c[j];
            }
            let mut cj = 0.0;
            for &i in slack_aux.basic.iter() {
                if i < num_nonbasic {
                    cj += standard.c[i] * slack_aux.a[i][j];
                }
            }
            c[j] -= cj;
        }

        let slack = SlackForm {
            nonbasic,
            basic,
            a,
            b,
            c,
            v,
        };
        return Some(slack);
    } else {
        return None;
    }
}

pub fn simplex(standard: &StandardForm) -> LPResult {
    let eps = 1e-8;

    // initialize basic solution
    let mut slack = match initialize_simplex(&standard) {
        Some(slack) => slack,
        None => return LPResult::Infeasible,
    };

    // pivoting
    while let Some(entering) = slack.c.iter().position(|&ci| ci > eps) {
        if slack.basic.iter().all(|&i| slack.a[i][entering] <= eps) { // Blant's rule
            return LPResult::Unbounded;
        }
        let mut leaving = 0;
        let mut delta = f64::INFINITY;
        for &ii in slack.basic.iter().filter(|&i| slack.a[*i][entering] > eps) { // Blant's rule
            let dii = slack.b[ii] / slack.a[ii][entering];
            if dii + eps < delta {
                delta = dii;
                leaving = ii;
            }
        }
        slack = slack.pivot(leaving, entering);
    }

    let basic_solution = slack.get_basic_solution();
    let optimal = slack.get_objective(&basic_solution);

    // convert to solution for standard form
    let num_nonbasic = slack.nonbasic.len();
    let mut solution = vec![0.0; num_nonbasic];
    for j in 0..num_nonbasic {
        solution[j] = basic_solution[j];
    }

    LPResult::Feasible((solution, optimal))
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate approx;
    use approx::{assert_relative_eq};

    #[test]
    fn test_feasible() {
        let c = vec![3.0, 1.0, 2.0];
        let a = vec![
            vec![1.0, 1.0, 3.0],
            vec![2.0, 2.0, 5.0],
            vec![4.0, 1.0, 2.0],
        ];
        let b = vec![30.0, 24.0, 36.0];
        let standard = StandardForm::new(c, a, b);

        match simplex(&standard) {
            LPResult::Feasible((solution, optimal)) => {
                assert_relative_eq!(solution[..], [8.0, 4.0, 0.0]);
                assert_relative_eq!(optimal, 28.0);
            },
            _ => unreachable!(),
        };

        let c = vec![10.0, 1.0];
        let b = vec![1.0, 100.0];
        let a = vec![
            vec![1.0, 0.0],
            vec![20.0, 1.0],
        ];
        let standard = StandardForm::new(c, a, b);
        match simplex(&standard) {
            LPResult::Feasible((solution, optimal)) => {
                assert_relative_eq!(solution[..], [0.0, 100.0]);
                assert_relative_eq!(optimal, 100.0);
            },
            _ => unreachable!(),
        };

        let c = vec![-2.0, -3.0, -1.0];
        let b = vec![-8.0, -6.0];
        let a = vec![
            vec![-1.0, -4.0, -2.0],
            vec![-3.0, -2.0, 0.0],
        ];
        let standard = StandardForm::new(c, a, b);
        match simplex(&standard) {
            LPResult::Feasible((solution, optimal)) => {
                assert_relative_eq!(solution[..], [0.8, 1.8, 0.0]);
                assert_relative_eq!(optimal, -7.0);
            },
            _ => unreachable!(),
        };

        let c = vec![1.0, -1.0, 1.0];
        let b = vec![4.0, -5.0, -1.0];
        let a = vec![
            vec![2.0, -1.0, 2.0],
            vec![2.0, -3.0, 1.0],
            vec![-1.0, 1.0, -2.0],
        ];
        let standard = StandardForm::new(c, a, b);
        match simplex(&standard) {
            LPResult::Feasible((solution, optimal)) => {
                assert_relative_eq!(solution[..], [0.0, 2.8, 3.4]);
                assert_relative_eq!(optimal, 0.6, epsilon = EPS);
            },
            _ => unreachable!(),
        };
    }

    #[test]
    fn test_unbounded() {
        let c = vec![1.0, 3.0, -1.0];
        let b = vec![10.0, 10.0, 10.0];
        let a = vec![
            vec![2.0, 2.0, -1.0],
            vec![3.0, -2.0, 1.0],
            vec![1.0, -3.0, 1.0],
        ];

        let standard = StandardForm::new(c, a, b);
        match simplex(&standard) {
            LPResult::Unbounded => (),
            _ => unreachable!(),
        };
    }

    #[test]
    fn test_infeasible() {
        let c = vec![3.0, 1.0];
        let b = vec![-1.0, -3.0, 2.0];
        let a = vec![
            vec![1.0, -1.0],
            vec![-1.0, -1.0],
            vec![2.0, 1.0],
        ];

        let standard = StandardForm::new(c, a, b);
        match simplex(&standard) {
            LPResult::Infeasible => (),
            _ => unreachable!(),
        };
    }

    #[test]
    fn test_initialize_simplex() {
        let c = vec![2.0, -1.0];
        let a = vec![
            vec![2.0, -1.0],
            vec![1.0, -5.0],
        ];
        let b = vec![2.0, -4.0];
        let standard = StandardForm::new(c, a, b);
        let slack = initialize_simplex(&standard).unwrap();

        let mut nonbasic = HashSet::new();
        nonbasic.insert(0);
        nonbasic.insert(3);
        assert_eq!(slack.nonbasic, nonbasic);

        let mut basic = HashSet::new();
        basic.insert(1);
        basic.insert(2);
        assert_eq!(slack.basic, basic);

        assert_relative_eq!(slack.v, -0.8);

        let c_expect = vec![1.8, 0.0, 0.0, -0.2];
        for &j in nonbasic.iter() {
            assert_relative_eq!(slack.c[j], c_expect[j]);
        }

        let b_expect = vec![0.0, 0.8, 2.8, 0.0];
        for &i in basic.iter() {
            assert_relative_eq!(slack.b[i], b_expect[i]);
        }

        let a_expect = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![-0.2, 0.0, 0.0, -0.2],
            vec![1.8, 0.0, 0.0, -0.2],
            vec![0.0, 0.0, 0.0, 0.0],
        ];
        for &i in basic.iter() {
            for &j in nonbasic.iter() {
                assert_relative_eq!(slack.a[i][j], a_expect[i][j]);
            }
        }
    }
}
