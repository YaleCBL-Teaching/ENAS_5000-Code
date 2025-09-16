import pytest
import sympy as sp
from ode_2nd_order_hw_symbolic import odes, solve_and_verify


# loop over all problems in the odes dictionary
@pytest.mark.parametrize("problem", odes.keys())
def test_symbolic_ode(problem):
    # solve_and_verify will throw an error if the solution is incorrect
    solve_and_verify(odes[problem])
