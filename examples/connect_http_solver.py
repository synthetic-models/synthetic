#!/usr/bin/env python3
"""
Example: Connecting to an HTTP-based ODE Simulation Server

This demonstrates how to use HTTPSolver to communicate with a remote
simulation server via HTTP. The server must implement the API specification
documented in docs/httpsolver_api.md.

Quick server setup (save as server.py and run with `uvicorn server:app --reload`):

    from fastapi import FastAPI
    from pydantic import BaseModel
    from typing import Dict, Optional

    app = FastAPI()

    class SimulationRequest(BaseModel):
        start: float
        stop: float
        step: float
        state_values: Optional[Dict[str, float]] = None
        parameter_values: Optional[Dict[str, float]] = None

    DEFAULT_STATES = {"S1": 100.0, "S1a": 0.0}
    DEFAULT_PARAMS = {"Vmax_J0": 100.0, "Km_J0": 50.0}

    @app.get("/states")
    async def get_states():
        return DEFAULT_STATES

    @app.get("/parameters")
    async def get_parameters():
        return DEFAULT_PARAMS

    @app.post("/simulate")
    async def simulate(req: SimulationRequest):
        return {
            "time": [req.start, req.stop],
            "S1": [100.0, 90.0],
            "S1a": [0.0, 10.0]
        }
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from synthetic.Solver.HTTPSolver import HTTPSolver


# Server configuration
SERVER_URL = "http://localhost:8000"


def main():
    print("HTTPSolver Example")
    print("-" * 40)

    # Create solver
    solver = HTTPSolver()

    # Compile with endpoint validation
    print(f"\n1. Connecting to {SERVER_URL}...")
    try:
        solver.compile(f"{SERVER_URL}/simulate")
        print("   Connected!")
    except ValueError as e:
        print(f"   Connection failed: {e}")
        print("   Make sure the server is running.")
        return

    # Get default states
    print("\n2. Fetching default states...")
    states = solver.get_state_defaults()
    print(f"   States: {states}")

    # Get default parameters
    print("\n3. Fetching default parameters...")
    params = solver.get_parameter_defaults()
    print(f"   Parameters: {params}")

    # Run simulation with defaults
    print("\n4. Running simulation (defaults)...")
    results = solver.simulate(start=0, stop=60, step=0.5)
    print(f"   Results:\n{results}")

    # Run simulation with state overrides
    print("\n5. Running simulation with state overrides...")
    solver.set_state_values({"Ribosome": 1000.0})
    results = solver.simulate(start=0, stop=60, step=0.5)
    print(f"   Results:\n{results}")

    # Run simulation with parameter overrides
    print("\n6. Running simulation with parameter overrides...")
    solver.set_parameter_values({"k1_1": 10.0})
    results = solver.simulate(start=0, stop=60, step=0.5)
    print(f"   Results:\n{results}")

    print("\nDone!")


if __name__ == "__main__":
    main()
