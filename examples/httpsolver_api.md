# HTTPSolver API Reference

This document describes the HTTP API contract for the `HTTPSolver` class. Server implementations should follow this specification to ensure compatibility.

## Overview

The `HTTPSolver` sends HTTP POST requests to an external server for ODE simulation. The server receives simulation parameters and optional state/parameter overrides, then returns simulation results as JSON.

## Client Usage

```python
from synthetic.Solver.HTTPSolver import HTTPSolver

solver = HTTPSolver()
solver.compile("http://localhost:8000/simulate", timeout=300.0)
# Note: compile() sends a HEAD request to validate the endpoint is reachable

# Get default state and parameter values from server
states = solver.get_state_defaults()
params = solver.get_parameter_defaults()
print(f"Default states: {states}")
print(f"Default parameters: {params}")

# Run simulation
results = solver.simulate(start=0, stop=1000, step=100)

# With overrides
solver.set_state_values({"S1": 100.0, "R1_1": 50.0})
solver.set_parameter_values({"Km_J0": 10.0})
results = solver.simulate(start=0, stop=1000, step=100)
```

**Important:** The `compile()` method validates the endpoint by sending a HEAD request. If the server is unreachable or returns an error status, a `ValueError` is raised with details about the connection failure.

---

## Server Implementation Guide

### Request Format

**Method:** `POST`

**Headers:**
```
Content-Type: application/json
```

**Body:**
```json
{
    "start": float,           // Required: Start time
    "stop": float,            // Required: Stop time
    "step": float,            // Required: Step size
    "state_values": {         // Optional: State name -> value overrides
        "S1": 100.0,
        "R1_1": 50.0
    },
    "parameter_values": {     // Optional: Parameter name -> value overrides
        "Km_J0": 10.0,
        "Vmax_J1": 100.0
    }
}
```

**Example request:**
```json
{
    "start": 0.0,
    "stop": 1000.0,
    "step": 100.0,
    "state_values": {
        "S1": 100.0,
        "R1_1": 50.0
    },
    "parameter_values": {
        "Km_J0": 10.0,
        "Vmax_J1": 100.0
    }
}
```

### Response Format

**Success:** `200 OK`

**Headers:**
```
Content-Type: application/json
```

The server should return JSON in one of these pandas.DataFrame-compatible formats. Prefer column-oriented format for efficiency.

#### Option A: Array of objects (row-oriented)
```json
[
    {"time": 0.0, "S1": 100.0, "S1a": 0.0, "R1_1": 50.0},
    {"time": 100.0, "S1": 95.0, "S1a": 5.0, "R1_1": 48.0},
    {"time": 200.0, "S1": 90.0, "S1a": 10.0, "R1_1": 46.0}
]
```

#### Option B: Object with arrays (column-oriented)
```json
{
    "time": [0.0, 100.0, 200.0],
    "S1": [100.0, 95.0, 90.0],
    "S1a": [0.0, 5.0, 10.0],
    "R1_1": [50.0, 48.0, 46.0]
}
```

### Error Responses

| Status Code | Description |
|-------------|-------------|
| `400 Bad Request` | Invalid parameters or malformed JSON |
| `500 Internal Server Error` | Simulation failure |
| `504 Gateway Timeout` | Simulation exceeded time limit |

**Error body format:**
```json
{
    "error": "Error message describing what went wrong"
}
```

### Endpoint Validation (HEAD Request)

When the client calls `compile()`, it sends a `HEAD` request to verify the endpoint is reachable. The server should handle HEAD requests gracefully:

| Status Code | Description |
|-------------|-------------|
| `200 OK` | Endpoint is valid and reachable |
| `404 Not Found` | Endpoint does not exist |
| `405 Method Not Allowed` | HEAD method not supported (client will treat as error) |

**Server implementation note:** If using FastAPI, HEAD requests are automatically handled for POST endpoints. For other frameworks, ensure the simulate endpoint responds to HEAD requests with a 200 status code, or implement an explicit HEAD handler.

### Model Info Endpoints

#### GET /states

Retrieve default initial values for all states (species) from the server.

**Request:**
```
GET /states
```

**Success:** `200 OK`

**Response:**
```json
{
    "Ribosome": 1900.0,
    "ppERK": 11.1,
    "ppAKT": 0.389,
    ...
}
```

**Error Responses:**
| Status Code | Description |
|-------------|-------------|
| `503 Service Unavailable` | Simulator not initialized |

**Example:**
```python
solver = HTTPSolver()
solver.compile("http://localhost:8000/simulate")
states = solver.get_state_defaults()
print(states)
# {'Ribosome': 1900.0, 'ppERK': 11.1, 'ppAKT': 0.389, ...}
```

#### GET /parameters

Retrieve default values for all fixed parameters from the server.

**Request:**
```
GET /parameters
```

**Success:** `200 OK`

**Response:**
```json
{
    "k1_1": 0.0042005,
    "k3_1": 140.7475,
    ...
}
```

**Error Responses:**
| Status Code | Description |
|-------------|-------------|
| `503 Service Unavailable` | Simulator not initialized |

**Example:**
```python
solver = HTTPSolver()
solver.compile("http://localhost:8000/simulate")
params = solver.get_parameter_defaults()
print(params)
# {'k1_1': 0.0042005, 'k3_1': 140.7475, ...}
```

### Behavior Notes

1. **Optional Overrides:** `state_values` and `parameter_values` are optional. If omitted, the server should use its default model configuration.

2. **Hot-Swappability:** The client clears overrides after each `simulate()` call. Each call with new overrides replaces the previous ones entirely.

3. **Server State Design:** The server can be either:
   - **Stateless:** Each request is independent; the server reloads the model for each request
   - **Stateful:** The server maintains the model state between requests (useful for cumulative effects)

---

## Example Server Implementation (Python/FastAPI)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, List
import pandas as pd

app = FastAPI()

class SimulationRequest(BaseModel):
    start: float
    stop: float
    step: float
    state_values: Optional[Dict[str, float]] = None
    parameter_values: Optional[Dict[str, float]] = None

class ErrorResponse(BaseModel):
    error: str

@app.get(
    "/states",
    tags=["Model Info"],
    responses={
        200: {"description": "Returns default state values"},
        503: {"model": ErrorResponse, "description": "Simulator not initialized"}
    }
)
async def get_states():
    """Get default initial values for all states (species)."""
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    return simulator.get_state_defaults()

@app.get(
    "/parameters",
    tags=["Model Info"],
    responses={
        200: {"description": "Returns default parameter values"},
        503: {"model": ErrorResponse, "description": "Simulator not initialized"}
    }
)
async def get_parameters():
    """Get default values for all fixed parameters."""
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    return simulator.get_parameter_defaults()

@app.post("/simulate")
async def simulate(req: SimulationRequest):
    try:
        # Run simulation with provided parameters
        results = run_simulation(
            start=req.start,
            stop=req.stop,
            step=req.step,
            state_values=req.state_values,
            parameter_values=req.parameter_values
        )
        # Return as DataFrame-compatible JSON (column-oriented)
        return results.to_dict(orient="list")
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})

# Note: FastAPI automatically handles HEAD requests for POST endpoints,
# so no additional handler is needed for endpoint validation.
```

---

## Example Server Implementation (Node.js/Express)

```javascript
const express = require('express');
const app = express();

app.use(express.json());

// GET /states - returns default state values
app.get('/states', (req, res) => {
    try {
        const stateDefaults = getStateDefaults(); // Your implementation
        res.json(stateDefaults);
    } catch (error) {
        res.status(503).json({ error: 'Simulator not initialized' });
    }
});

// GET /parameters - returns default parameter values
app.get('/parameters', (req, res) => {
    try {
        const paramDefaults = getParameterDefaults(); // Your implementation
        res.json(paramDefaults);
    } catch (error) {
        res.status(503).json({ error: 'Simulator not initialized' });
    }
});

// HEAD handler for endpoint validation (called by client during compile)
app.head('/simulate', (req, res) => {
    res.status(200).end();
});

app.post('/simulate', (req, res) => {
    const { start, stop, step, state_values, parameter_values } = req.body;

    try {
        const results = runSimulation({
            start,
            stop,
            step,
            stateValues: state_values,
            parameterValues: parameter_values
        });

        res.json(results);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.listen(8000);
```
