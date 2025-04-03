import json
import pennylane as qml
import pennylane.numpy as np
# Write any helper functions you need here

def cost_hamiltonian(edges):
    """
    Constructs the cost Hamiltonian for the Minimum Vertex Cover problem.

    Args:
        edges (list[list[int]]): List of edges in the graph.

    Returns:
        pennylane.Hamiltonian: The cost Hamiltonian.
    """
    # Get the unique vertices
    u_vertices = set()
    for edge in edges:
        u_vertices.update(edge)
    u_vertices = list(u_vertices)

    # Compute the second term of the Hamiltonian
    one_coeff = [1] * len(u_vertices)
    obs = [qml.PauliZ(i) for i in u_vertices]
    H_second = qml.dot(one_coeff, obs)

    # Compute the first term of the Hamiltonian
    one_coeff = [1] * len(edges)
    obs = [qml.PauliZ(i) @ qml.PauliZ(j) + qml.PauliZ(i) + qml.PauliZ(j) for i, j in edges]
    H_first = qml.dot(one_coeff, obs)

    return (3 / 4) * H_first - H_second

def mixer_hamiltonian(edges):
    """
    Constructs the mixer Hamiltonian for the QAOA algorithm.

    Args:
        edges (list[list[int]]): List of edges in the graph.

    Returns:
        pennylane.Hamiltonian: The mixer Hamiltonian.
    """
    # Get the unique vertices
    u_vertices = set()
    for edge in edges:
        u_vertices.update(edge)
    u_vertices = list(u_vertices)

    # Compute the mixer Hamiltonian
    one_coeff = [1] * len(u_vertices)
    obs = [qml.PauliX(i) for i in u_vertices]
    H_mixer = qml.dot(one_coeff, obs)

    return H_mixer

def qaoa_layer(h_cost, h_mixer, gamma, alpha):
    """
    A single QAOA layer consisting of cost and mixer Hamiltonians.

    Args:
        h_cost (qml.Hamiltonian): Cost Hamiltonian.
        h_mixer (qml.Hamiltonian): Mixer Hamiltonian.
        gamma (float): Parameter for the cost Hamiltonian.
        alpha (float): Parameter for the mixer Hamiltonian.
    """
    qml.ApproxTimeEvolution(h_cost, gamma, 1)
    qml.ApproxTimeEvolution(h_mixer, alpha, 1)


def qaoa_circuit(params, edges):
    """
    Implements the QAOA circuit.

    Args:
        params (np.array): QAOA parameters.
        edges (list[list[int]]): List of edges in the graph.
    """
    depth = 4 # Number of QAOA layers

    h_cost = cost_hamiltonian(edges)
    h_mixer = mixer_hamiltonian(edges)

    # Apply Hadamard gates to all qubits
    u_vertices = set()
    for edge in edges:
        u_vertices.update(edge)
    u_vertices = list(u_vertices)

    wires = range(len(u_vertices))

    for w in wires:
        qml.Hadamard(wires=w)

    # Apply QAOA layers
    qml.layer(qaoa_layer, depth, [h_cost, h_cost, h_cost, h_cost], [h_mixer, h_mixer, h_mixer, h_mixer], params[0], params[1])

# This function runs the QAOA circuit and returns the expectation value of the cost Hamiltonian

dev = qml.device("default.qubit")

@qml.qnode(dev)
def qaoa_expval(params, edges):
    qaoa_circuit(params, edges)
    return qml.expval(cost_hamiltonian(edges))

def optimize(edges):
    """
    Optimizes the QAOA parameters.

    Args:
        edges (list[list[int]]): List of edges in the graph.

    Returns:
        np.array: Optimized QAOA parameters.
    """
    opt = qml.GradientDescentOptimizer()
    steps = 300

    # Initialize parameters
    depth = 4  # Number of QAOA layers
    params = np.array([[0.5] * depth, [0.5] * depth], requires_grad=True)

    # Optimize parameters
    for _ in range(steps):
        params = opt.step(lambda v: qaoa_expval(v, edges), params)

    return params

# These are auxiliary functions that will help us grade your solution. Feel free to check them out!

@qml.qnode(dev)
def qaoa_probs(params, edges):
  qaoa_circuit(params, edges)
  return qml.probs()

def approximation_ratio(params, edges):
    """
    Calculates the approximation ratio of the QAOA solution.

    Args:
        params (np.array): QAOA parameters.
        edges (list[list[int]]): List of edges in the graph.

    Returns:
        float: Approximation ratio.
    """
    true_min = np.min(qml.eigvals(cost_hamiltonian(edges)))

    # Handle cases where the true minimum eigenvalue is zero
    if np.isclose(true_min, 0):
        print("Warning: The true minimum eigenvalue of the cost Hamiltonian is zero.")
        return 0.0

    approx_ratio = qaoa_expval(params, edges) / true_min
    return approx_ratio

# These functions are responsible for testing the solution.

def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    params = optimize(ins)
    output= approximation_ratio(params,ins)

    ground_energy = np.min(qml.eigvals(cost_hamiltonian(ins)))

    index = np.argmax(qaoa_probs(params, ins))
    vector = np.zeros(len(qml.matrix(cost_hamiltonian(ins))))
    vector[index] = 1

    calculate_energy = np.real_if_close(np.dot(np.dot(qml.matrix(cost_hamiltonian(ins)), vector), vector))
    verify = np.isclose(calculate_energy, ground_energy)

    if verify:
      return str(output)
    
    return "QAOA failed to find right answer"

def check(solution_output: str, expected_output: str) -> None:

    assert not solution_output == "QAOA failed to find right answer", "QAOA failed to find the ground eigenstate."
        
    print(f"Solution output: {solution_output}")
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert solution_output >= expected_output-0.01, "Minimum approximation ratio not reached"


# These are the public test cases
test_cases = [
    ('[[0, 1], [1, 2], [0, 2], [2, 3]]', '0.55'),
    ('[[0, 1], [1, 2], [2, 3], [3, 0]]', '0.92'),
    ('[[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 4]]', '0.55')
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