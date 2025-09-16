#!/usr/bin/env python3

# Solve y'' + 2 y' - 3 y = 8 e^x symbolically with SymPy
import sympy as sp

x = sp.symbols('x')
y = sp.Function('y')

# Define the left and right hand sides of the ODE
ode = sp.Eq(sp.diff(y(x), x, 2) + 2*sp.diff(y(x), x) - 3*y(x), 8*sp.exp(x))

# Solve the ODE
sol = sp.dsolve(ode)

# Verify the solution by substituting back into the left-hand side
lhs = sp.simplify(sp.diff(sol.rhs, x, 2) + 2*sp.diff(sol.rhs, x) - 3*sol.rhs)
check = sp.simplify(lhs - 8*sp.exp(x))

print("General solution y(x):")
sp.pprint(sol)
print("\nVerification (should be 0):", check)
sp.pprint(check)
