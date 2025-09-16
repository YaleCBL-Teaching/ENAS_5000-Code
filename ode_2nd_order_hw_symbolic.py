#!/usr/bin/env python3

import sympy as sp

t = sp.symbols("t", real=True)
y = sp.Function("y")

# Dictionary containing the ODEs with their problem numbers as keys
odes = {
    "homogeneous 01": sp.Eq(sp.diff(y(t), t, 2) - sp.diff(y(t), t) - 6 * y(t), 0),
    "homogeneous 02": sp.Eq(sp.diff(y(t), t, 2) - 2 * sp.diff(y(t), t) + 10 * y(t), 0),
    "homogeneous 03": sp.Eq(sp.diff(y(t), t, 2) + 6 * sp.diff(y(t), t) + 9 * y(t), 0),
    "homogeneous 04": sp.Eq(sp.diff(y(t), t, 2) - 3 * sp.diff(y(t), t), 0),
    "nonhomogeneous 07": sp.Eq(
        sp.diff(y(t), t, 2) - sp.diff(y(t), t) - 2 * y(t), 2 * t**2 + 5
    ),
    "nonhomogeneous 10": sp.Eq(
        sp.diff(y(t), t, 2) - 4 * sp.diff(y(t), t) + 5 * y(t), 21 * sp.exp(2 * t)
    ),
    "nonhomogeneous 12": sp.Eq(
        sp.diff(y(t), t, 2) + 6 * sp.diff(y(t), t) + 9 * y(t), 9 * sp.cos(3 * t)
    ),
}


def solve_and_verify(ode_eq):
    # solve the ODE
    sol = sp.dsolve(ode_eq)

    # uncomment this line to make the tests fail
    #sol = sp.Eq(y(t), sol.rhs + sp.sin(t))

    # verify the solution by substituting back into the left-hand side
    ok, resid = sp.checkodesol(ode_eq, sol)
    if not ok:
        raise ValueError(f"Solution failed with residual: {sp.simplify(resid)}")
    return sol


if __name__ == "__main__":
    # iterate over the dictionary items
    for problem, equation in odes.items():
        sol = solve_and_verify(equation)
        print(f"Problem {problem}: y(t) = {sp.simplify(sol.rhs)}")
