from models.Solver.Solver import Solver

import re
from typing import List, Dict, Any, Tuple

import pandas as pd
from scipy.integrate import odeint
import sympy as sp
import numpy as np
from numba import njit

class ScipySolver(Solver):
    """
    Solver using scipy to solve the ODEs. 
    """
    
    def __init__(self):
        super().__init__()
        # reactions, species, parameters, y0, parameter_values
        self.reactions = None
        self.species = None
        self.parameters = None
        self.y0 = None
        self.parameter_values = None
        self.assignment_rules = None
        self.func = None
        self.jit = False
        
        self.last_sim_result = None
        
    def compile(self, compile_str: str, **kwargs) -> None:
        """
        Compile the model from an Antimony string and generate ODE function.
        Assignment rule variables are treated like pseudo-parameters with
        time-based value changes during simulation.
        """
        result = self._parse_antimony_model(compile_str)
        reactions, species, parameters, y0, parameter_values, assignment_rules = result

        self.reactions = reactions
        self.species = species
        self.assignment_rules = assignment_rules

        # Extend parameters with assignment rule variables
        self.assignment_rule_vars = sorted(assignment_rules.keys())
        self.parameters = parameters            # already includes rule vars
        self.parameter_values = parameter_values  
        self.y0 = y0

        self.jit = kwargs.get("jit", False)
        if not self.jit:
            self.func = self._reactions_to_ode_func(
                reactions, species, self.parameters, assignment_rule_vars=self.assignment_rule_vars
            )
        else:
            self.func = self._reactions_to_jit_ode_func(
                reactions, species, self.parameters, assignment_rule_vars=self.assignment_rule_vars
            )
    
        
    def simulate(self, start: float, stop: float, step: float) -> pd.DataFrame:
        """
        Simulate the ODE system from start to stop with a given step size.
        Handles piecewise assignment rule changes at change-points.
        """
        if self.func is None:
            raise ValueError("Model instance is not created. Please call compile() first.")

        # Generate full time vector and extract rule breakpoints
        t_full = np.linspace(start, stop, step)
        change_points = sorted(
            {rule["time"] for rule in self.assignment_rules.values() if start < rule["time"] < stop}
        )
        time_segments = [start] + change_points + [stop]

        all_results = []
        y_current = np.array(self.y0, dtype=float)
        param_current = list(self.parameter_values)
        param_map = {p: i for i, p in enumerate(self.parameters)}

        for i in range(len(time_segments) - 1):
            t0, t1 = time_segments[i], time_segments[i + 1]
            mask = (t_full >= t0) & (t_full <= t1)
            t_segment = np.unique(t_full[mask])  # Remove duplicate times

            if len(t_segment) < 2:
                continue  # Skip short segments

            # Safety checks
            if not np.all(np.isfinite(y_current)):
                raise ValueError(f"Non-finite state values before integrating [{t0}, {t1}]: {y_current}")
            if not np.all(np.isfinite(param_current)):
                raise ValueError(f"Non-finite parameter values before integrating [{t0}, {t1}]: {param_current}")

            try:
                if self.jit:
                    def ode_wrapper(y, t, *args):
                        return self.func(y, t, np.array(args))
                    sol = odeint(ode_wrapper, y_current, t_segment, args=tuple(param_current))
                else:
                    sol = odeint(self.func, y_current, t_segment, args=tuple(param_current))
            except Exception as e:
                raise RuntimeError(f"ODE integration failed between t={t0} and t={t1}: {e}")

            # Store segment result
            df_segment = pd.DataFrame(sol, columns=self.species)
            df_segment.insert(0, "time", t_segment)
            all_results.append(df_segment)

            # Update state for next segment
            y_current = sol[-1]

            # Apply assignment rule changes at t1
            for var, rule in self.assignment_rules.items():
                if np.isclose(rule["time"], t1):
                    idx = param_map.get(var)
                    if idx is None:
                        raise KeyError(f"Assignment rule targets unknown parameter '{var}'")
                    param_current[idx] = rule["after"]
                    # Optional debug log:
                    # print(f"At t={t1}, setting {var} = {rule['after']}")

        final_result = pd.concat(all_results, ignore_index=True)
        self.last_sim_result = final_result
        return final_result




    def set_state_values(self, state_values: Dict[str, float]) -> bool:
        """
        Hot swapping of state variables in the running instance of the model, note this is setting the initial values of the state variables.
        Set the values of state variables in the model instance, this should only possible after compiling the model. 
        Not every solver will support this, so it is possible that this function to return an not implemented error.
        returns True if the state variable was set successfully, False otherwise.
        """
        # Check if the model is created
        if self.func is None:
            raise ValueError("Model instance is not created. Please call compile() first.")
        
        # Check if the state values are valid
        for key in state_values.keys():
            if key not in self.species:
                raise ValueError(f"State variable {key} is not valid. Valid state variables are: {self.species}")
        
        # Set the state values
        for key, value in state_values.items():
            index = self.species.index(key)
            self.y0[index] = value
        
        return True

    def set_parameter_values(self, parameter_values: Dict[str, float]) -> bool:
        """
        Hot swapping of parameters in the running instance of the model.
        Set the values of parameter variables in the model instance, this should only possible after compiling the model. 
        Not every solver will support this, so it is possible that this function to return an not implemented error.
        returns True if the state variable was set successfully, False otherwise.
        """
        # Check if the model is created
        if self.func is None:
            raise ValueError("Model instance is not created. Please call compile() first.")
        
        # Check if the parameter values are valid
        for key in parameter_values.keys():
            if key not in self.parameters:
                raise ValueError(f"Parameter {key} is not valid. Valid parameters are: {self.parameters}")
        
        # Set the parameter values
        for key, value in parameter_values.items():
            index = self.parameters.index(key)
            self.parameter_values[index] = value
        
        return True
        
    
    def _parse_antimony_model(self, antimony_str: str) -> Tuple[
        List[str], List[str], List[str], List[float], List[float], Dict[str, Dict[str, float]]
    ]:
        """
        Parses an Antimony model string and extracts:
        - reactions
        - species
        - parameters
        - initial conditions (y0)
        - parameter values
        - assignment rules as dicts of {var: {'before': val, 'after': val, 'time': t}}
        """
        reactions = []
        species_set = set()
        species_dict = {}
        parameter_dict = {}
        assignment_rules = {}

        for line in antimony_str.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("model ") or line == "end":
                continue

            # Reaction line
            if ":" in line and ";" in line:
                _, rxn = line.split(":", 1)
                rxn = rxn.strip()
                reactions.append(rxn)

                reaction_part, _ = map(str.strip, rxn.split(";"))
                if "->" in reaction_part:
                    lhs, rhs = map(str.strip, reaction_part.split("->"))
                    reactants = [s.strip() for s in lhs.split("+")]
                    products = [s.strip() for s in rhs.split("+")]
                    species_set.update(reactants + products)

            # Assignment rule
            elif ":=" in line:
                var, expr = map(str.strip, line.split(":=", 1))
                rule = self._parse_piecewise_rule(expr)
                if rule:
                    assignment_rules[var] = rule

            # Initial value
            elif "=" in line:
                var, val = map(str.strip, line.split("=", 1))
                try:
                    val = float(val)
                    if var in species_set:
                        species_dict[var] = val
                    else:
                        parameter_dict[var] = val
                except ValueError:
                    continue  # skip non-numeric initial values

        species = sorted(species_dict)
        y0 = [species_dict[s] for s in species]
        parameters = sorted(parameter_dict)
        parameter_values = [parameter_dict[p] for p in parameters]

        # Treat assignment rule names as "pseudo-parameters"
        parameters += sorted(assignment_rules.keys())
        parameter_values += [rule['before'] for rule in assignment_rules.values()]
        return reactions, species, parameters, y0, parameter_values, assignment_rules


    def _parse_piecewise_rule(self, expr: str) -> Dict[str, float] | None:
        """
        Parses simple piecewise expressions like:
        'piecewise(0, time < 500, 500)' → {'before': 0.0, 'after': 500.0, 'time': 500.0}
        """
        match = re.match(r"piecewise\(([^,]+),\s*time\s*<\s*([0-9.]+),\s*([^)]+)\)", expr)
        if match:
            val1, time_str, val2 = match.groups()
            try:
                return {
                    "before": float(val1),
                    "after": float(val2),
                    "time": float(time_str)
                }
            except ValueError:
                return None
        return None

    
    def _reactions_to_ode_func(self, reactions, species, parameters, assignment_rule_vars=None):
        """
        Convert a list of reactions to an ODE function using sympy.
        Supports symbolic variables for both parameters and assignment rules.
        """
        species_syms = {s: sp.Symbol(s) for s in species}
        param_syms = {p: sp.Symbol(p) for p in parameters}
        rule_syms = {r: sp.Symbol(r) for r in assignment_rule_vars or []}

        all_syms = {**species_syms, **param_syms, **rule_syms}

        derivs = {s: 0 for s in species}

        for rxn in reactions:
            reaction_part, rate_expr = map(str.strip, rxn.split(";"))
            rate = sp.sympify(rate_expr, locals=all_syms)

            if "<->" in reaction_part:
                raise NotImplementedError("Reversible reactions not yet supported.")

            reactants_str, products_str = map(str.strip, reaction_part.split("->"))
            reactants = [r.strip() for r in reactants_str.split("+") if r.strip()]
            products = [p.strip() for p in products_str.split("+") if p.strip()]

            for r in reactants:
                derivs[r] -= rate
            for p in products:
                derivs[p] += rate

        dydt_exprs = [derivs[s] for s in species]

        # 🟢 Correct: single flat argument list
        all_inputs = list(species_syms.values()) + list(param_syms.values())
        dydt_func = sp.lambdify(all_inputs, dydt_exprs, modules="numpy")

        def func(y, t, *params):
            return np.array(dydt_func(*y, *params)).flatten()

        return func




    def _reactions_to_jit_ode_func(self, reactions, species, parameters, assignment_rule_vars=None):
        """
        Create a Numba-compiled ODE function that supports assignment rules as parameters.
        """
        # Create symbols
        species_syms = {s: sp.Symbol(s) for s in species}
        param_syms = {p: sp.Symbol(p) for p in parameters}
        rule_syms = {r: sp.Symbol(r) for r in assignment_rule_vars or []}
        all_syms = {**species_syms, **param_syms, **rule_syms}

        # Derivative dictionary
        derivs = {s: 0 for s in species}

        for rxn in reactions:
            reaction_part, rate_expr = map(str.strip, rxn.split(";"))
            rate = sp.sympify(rate_expr, locals=all_syms)

            if "<->" in reaction_part:
                raise NotImplementedError("Reversible reactions not yet supported.")

            reactants_str, products_str = map(str.strip, reaction_part.split("->"))
            reactants = [r.strip() for r in reactants_str.split("+") if r.strip()]
            products = [p.strip() for p in products_str.split("+") if p.strip()]

            for r in reactants:
                derivs[r] -= rate
            for p in products:
                derivs[p] += rate

        dydt_exprs = [derivs[s] for s in species]

        # Generate symbolic subs
        species_subs = {species_syms[s]: sp.Symbol(f"y[{i}]") for i, s in enumerate(species)}
        param_subs = {param_syms[p]: sp.Symbol(f"params[{i}]") for i, p in enumerate(parameters)}

        func_lines = ["def generated_func(y, t, params):"]
        func_lines.append(f"    dydt = np.empty({len(dydt_exprs)})")

        for i, expr in enumerate(dydt_exprs):
            substituted = expr.subs({**species_subs, **param_subs})
            code_line = sp.ccode(substituted)
            func_lines.append(f"    dydt[{i}] = {code_line}")

        func_lines.append("    return dydt")

        func_code = "\n".join(func_lines)

        local_vars = {"np": np}
        exec(func_code, local_vars)
        generated_func = local_vars["generated_func"]

        return njit(generated_func)
