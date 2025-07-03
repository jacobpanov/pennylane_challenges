import json
import pennylane as qml
import pennylane.numpy as np
edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
num_wires = 4

# We define the Hamiltonian for you!

ops = [qml.PauliZ(0), qml.PauliZ(1),qml.PauliZ(2), qml.PauliZ(3), qml.PauliZ(0)@qml.PauliZ(1), qml.PauliZ(0)@qml.PauliZ(2),qml.PauliZ(1)@qml.PauliZ(2),qml.PauliZ(2)@qml.PauliZ(3)]
coeffs = [0.5, 0.5, 1.25, -0.25, 0.75, 0.75, 0.75, 0.75]

cost_hamiltonian = qml.Hamiltonian(coeffs, ops)

# Write any helper functions you need here

dev = qml.device("default.mixed", wires=num_wires)

@qml.qnode(dev)
def _qaoa_circuit(params, noise_param):
    """Noisy QAOA circuit implemented with native rotations and noisy CNOTs.

    Args:
        params (list[list[float]]): QAOA parameters ``[[gamma_0, beta_0], ...]``.
        noise_param (float): Depolarizing probability added after each CNOT.

    Returns:
        qml.numpy.tensor: Expectation value of the cost Hamiltonian.
    """

    # Prepare the ``|+>`` state on all qubits using rotation gates only
    for w in range(num_wires):
        qml.RY(np.pi / 2, wires=w)

    # Iterate over the QAOA layers
    for gamma, beta in params:
        # ----- Cost Hamiltonian terms -----
        # Single Z rotations
        qml.RZ(2 * gamma * 0.5, wires=0)
        qml.RZ(2 * gamma * 0.5, wires=1)
        qml.RZ(2 * gamma * 1.25, wires=2)
        qml.RZ(2 * gamma * -0.25, wires=3)

        # ZZ interactions. Each CNOT is followed by a depolarizing channel.
        for control, target in [(0, 1), (0, 2), (1, 2), (2, 3)]:
            qml.CNOT(wires=[control, target])
            qml.DepolarizingChannel(noise_param, wires=target)
            qml.RZ(2 * gamma * 0.75, wires=target)
            qml.CNOT(wires=[control, target])
            qml.DepolarizingChannel(noise_param, wires=target)

        # ----- Mixer Hamiltonian -----
        for w in range(num_wires):
            qml.RX(2 * beta, wires=w)

    return qml.expval(cost_hamiltonian)


def qaoa_circuit(params, noise_param):
    """Wrapper for ``_qaoa_circuit`` that stores the executed tape."""
    res = _qaoa_circuit(params, noise_param)
    if hasattr(_qaoa_circuit, "_tape") and _qaoa_circuit._tape is not None:
        qaoa_circuit.qtape = _qaoa_circuit._tape
    return res


def approximation_ratio(qaoa_depth, noise_param):
    """
    Returns the approximation ratio of the QAOA algorithm for the Minimum Vertex Cover of the given graph
    with depolarizing gates after each native CNOT gate.

    Args:
        qaoa_depth (int): The number of cost/mixer layers in the QAOA algorithm used.
        noise_param (float): The noise parameter associated with the depolarization gate.

    Returns: 
        (float): The approximation ratio for the noisy QAOA.
    """
    # random initialization of the variational parameters
    params = np.random.uniform(0, 2 * np.pi, (qaoa_depth, 2))

    # cost function to minimize
    def cost(p):
        return qaoa_circuit(p, noise_param)

    opt = qml.AdamOptimizer(0.1)
    steps = 200 if qaoa_depth == 1 else 300

    for _ in range(steps):
        params = opt.step(cost, params)

    noisy_min_expval = qaoa_circuit(params, noise_param)

    true_min = np.min(np.linalg.eigvalsh(cost_hamiltonian.sparse_matrix().todense()))

    return noisy_min_expval / true_min


# These functions are responsible for testing the solution.
random_params = np.array([np.random.rand(2)])

ops_2 = [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2), qml.PauliX(3)]
coeffs_2 = [1,1,1,1]

mixer_hamiltonian = qml.Hamiltonian(coeffs_2, ops_2)

@qml.qnode(dev)
def noiseless_qaoa(params):

    for wire in range(num_wires):

        qml.Hadamard(wires = wire)

    for elem in params:

        qml.ApproxTimeEvolution(cost_hamiltonian, elem[0], 1)
        qml.ApproxTimeEvolution(mixer_hamiltonian, elem[1],1)

    return qml.expval(cost_hamiltonian)

random_params = np.array([np.random.rand(2)])

circuit_check = (np.isclose(noiseless_qaoa(random_params) - qaoa_circuit(random_params,0),0))

def run(test_case_input: str) -> str:
    input = json.loads(test_case_input)
    output = approximation_ratio(*input)

    return str(output)

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    
    tape = qaoa_circuit.qtape
    names = [op.name for op in tape.operations]
    random_params = np.array([np.random.rand(2)])

    assert circuit_check, "qaoa_circuit is not doing what it's expected to."

    assert names.count('ApproxTimeEvolution') == 0, "Your circuit must not use the built-in PennyLane Trotterization."
     
    assert set(names) == {'DepolarizingChannel', 'RX', 'RY', 'RZ', 'CNOT'}, "Your circuit must use qml.RX, qml.RY, qml.RZ, qml.CNOT, and qml.DepolarizingChannel."

    assert solution_output > expected_output - 0.02

# These are the public test cases
test_cases = [
    ('[2,0.005]', '0.4875'),
    ('[1, 0.003]', '0.1307')
]
# This will run the public test cases locally
for i, (input_, expected_output) in enumerate(test_cases):
    print(f"Running test case {i} with input '{input_}'...")

    try:
        output = run(input_)

    except Exception as exc:
        print(f"Runtime Error. {exc}")

    else:
        if message := check(output, expected_output):
            print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")

        else:
            print("Correct!")
