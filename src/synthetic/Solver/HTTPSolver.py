# HTTP Solver
# This class sends HTTP requests to an external server for ODE simulation

from typing import Dict, Optional
import pandas as pd
import requests
from .Solver import Solver


class HTTPSolver(Solver):
    """
    A solver that sends HTTP POST requests to an external server for ODE simulation.

    The server receives simulation parameters (start, stop, step) and optional
    state/parameter overrides via JSON, and returns results as JSON.

    Server Implementation Guide:
        See docs/httpsolver_api.md for complete API specification including:
        - Request/response formats
        - Error handling
        - Example server implementations (FastAPI, Node.js/Express)

    Example:
        solver = HTTPSolver()
        solver.compile("http://localhost:8000/simulate")
        solver.set_state_values({"S1": 100.0})
        results = solver.simulate(0, 1000, 100)
    """

    def __init__(self):
        super().__init__()
        self.endpoint: Optional[str] = None
        self.timeout: float = 300.0  # Default timeout for requests (seconds)
        self.state_values: Dict[str, float] = {}
        self.param_values: Dict[str, float] = {}

    def compile(self, compile_str: str, **kwargs) -> bool:
        """
        Compile the model by storing the HTTP endpoint URL.

        Args:
            compile_str: The HTTP endpoint URL (e.g., "http://localhost:8000/simulate")
            **kwargs: Additional arguments:
                - timeout: Request timeout in seconds (default: 300.0)
                - headers: Dict of HTTP headers to include in requests
                - auth: Tuple of (username, password) for basic auth

        Returns:
            True
        """
        self.endpoint = compile_str.rstrip('/')  # Remove trailing slash if present
        self.timeout = kwargs.get('timeout', 300.0)
        self.headers = kwargs.get('headers', {})
        self.auth = kwargs.get('auth')

        return True

    def simulate(self, start: float, stop: float, step: float) -> pd.DataFrame:
        """
        Simulate by sending a POST request to the HTTP endpoint.

        Args:
            start: Start time of the simulation
            stop: Stop time of the simulation
            step: Step size for the simulation

        Returns:
            pandas DataFrame with 'time' column and species columns

        Raises:
            requests.RequestException: If the HTTP request fails
            ValueError: If the response is invalid
        """
        if self.endpoint is None:
            raise RuntimeError("Solver not compiled. Call compile() with an endpoint URL first.")

        # Prepare JSON payload
        payload = {
            'start': start,
            'stop': stop,
            'step': step
        }

        # Include state and parameter values if they have been set
        if self.state_values:
            payload['state_values'] = self.state_values
        if self.param_values:
            payload['parameter_values'] = self.param_values

        # Prepare headers
        headers = self.headers.copy()
        headers['Content-Type'] = 'application/json'

        # Send POST request
        response = requests.post(
            self.endpoint,
            json=payload,
            timeout=self.timeout,
            headers=headers,
            auth=self.auth
        )
        response.raise_for_status()

        # Clear state and parameter values after sending (hot swappability)
        self.state_values.clear()
        self.param_values.clear()

        # Parse JSON response
        try:
            data = response.json()
        except ValueError as e:
            raise ValueError(f"Invalid JSON response from server: {e}")

        # Convert to DataFrame
        try:
            df = pd.DataFrame(data)
            self.last_sim_result = df
            return df
        except Exception as e:
            raise ValueError(f"Could not convert response to DataFrame: {e}")

    def set_state_values(self, state_values: Dict[str, float]) -> bool:
        """
        Set state values to be sent in the next simulate() call.

        The values replace any previously set state values and are included
        in the payload of the next simulate() request. Values are cleared
        after the request completes, enabling hot-swappability.

        Args:
            state_values: Dictionary mapping state names to their values

        Returns:
            True if the state values were set successfully
        """
        self.state_values = state_values.copy()
        return True

    def set_parameter_values(self, parameter_values: Dict[str, float]) -> bool:
        """
        Set parameter values to be sent in the next simulate() call.

        The values replace any previously set parameter values and are included
        in the payload of the next simulate() request. Values are cleared
        after the request completes, enabling hot-swappability.

        Args:
            parameter_values: Dictionary mapping parameter names to their values

        Returns:
            True if the parameter values were set successfully
        """
        self.param_values = parameter_values.copy()
        return True
