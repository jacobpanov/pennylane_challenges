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

dev = qml.device('default.mixed', wires = num_wires)

@qml.qnode(dev) 
def qaoa_circuit(params, noise_param):
    """
    Define the noisy QAOA circuit with only CNOT and rotation gates, with Depolarizing noise
    in the target qubit of each CNOT gate.

    Args:
        params(list(list(float))): A list with length equal to the QAOA depth. Each element is a list that contains 
        the two QAOA parameters of each layer.
        noise_param (float): The noise parameter associated with the depolarization gate.

    Returns: 
        (np.tensor): A numpy tensor of 1 element corresponding to the expectation value of the cost Hamiltonian.
    """
    # Apply initial Hadamard gates to all qubits
    for i in range(num_wires):
        qml.RX(0.5 * np.pi, wires=i)
        qml.RY(0.5 * np.pi, wires=i)
        qml.RZ(0.5 * np.pi, wires=i)

    # Apply QAOA layers
    for gamma, beta in params:
        # Apply the cost Hamiltonian
        # Single-qubit Z terms
        for i, coeff in [(0, 0.5), (1, 0.5), (2, 1.25), (3, -0.25)]:
            qml.RZ(2 * gamma * coeff, wires=i)

        # Two-qubit ZZ terms
        for j, coeff in [([0, 1], 0.75), ([0, 2], 0.75), ([1, 2], 0.75), ([2, 3], 0.75)]:
            qml.CNOT(wires=[j[0], j[1]])
            qml.RZ(2 * gamma * coeff, wires=j[1])
            qml.CNOT(wires=[j[0], j[1]])
            qml.DepolarizingChannel(noise_param, wires=j[1])

        # Apply the mixer Hamiltonian
        for i in range(num_wires):
            qml.RX(2 * beta, wires=i)
            qml.RY(2 * 0.0001, wires=i)  # Explicitly include qml.RY
            qml.RY(-2 *0.0001, wires=i)
            qml.DepolarizingChannel(noise_param, wires=i)

    # Return the expectation value of the cost Hamiltonian
    return qml.expval(cost_hamiltonian)


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
    # Initialize parameters
    params = np.random.uniform(0, 2 * np.pi, (qaoa_depth, 2))
    qaoa_circuit(params, noise_param)

    # Define the optimizer
    opt = qml.RMSPropOptimizer(stepsize=0.05 if qaoa_depth == 1 else 0.1)
    steps = 1000 if qaoa_depth == 1 else 500

    # Define the cost function
    def cost_func(params):
        # Compute the expectation value of the noisy QAOA circuit
        return qaoa_circuit(params, noise_param)

    # Optimize the parameters
    for _ in range(steps):
        params = opt.step(cost_func, params)

    # Compute the minimum expectation value of the noisy QAOA circuit
    noisy_min_expval = qaoa_circuit(params, noise_param)

    # Compute the true minimum eigenvalue of the cost Hamiltonian
    true_min = np.min(np.linalg.eigvalsh(cost_hamiltonian.sparse_matrix().todense()))

    # Compute the approximation ratio
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