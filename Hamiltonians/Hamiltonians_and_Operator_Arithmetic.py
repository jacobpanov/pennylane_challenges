import json
import pennylane as qml
import pennylane.numpy as np
def hamiltonian(num_wires):
    """
    A function for creating the Hamiltonian in question for a general
    number of qubits.

    Args:
        num_wires (int): The number of qubits.

    Returns:
        (qml.Hamiltonian): A PennyLane Hamiltonian.
    """

    # Coefficients and operators for the Hamiltonian
    coeffs = []
    ops = []

    coeffs = [1/3] * (num_wires * (num_wires - 1) // 2) + [-1] * num_wires
    # Add the two-qubit Z ⊗ Z terms
    num_pairs = num_wires * (num_wires - 1) / 2  # Total number of pairs
    for i in range(num_wires):
        for j in range(i + 1, num_wires):
            #coeffs.append(1 / num_pairs)  # Normalized coefficient for Z_i Z_j
            ops.append(qml.PauliX(i) @ qml.PauliX(j))

    # Add the single-qubit X terms
    for i in range(num_wires):
        #coeffs.append(1)  # Coefficient for X_i
        ops.append(qml.PauliZ(i))

    
    # Return the Hamiltonian
    return qml.Hamiltonian(coeffs, ops)

def expectation_value(num_wires):
    """
    Simulates the circuit in question and returns the expectation value of the 
    Hamiltonian in question.

    Args:
        num_wires (int): The number of qubits.

    Returns:
        (float): The expectation value of the Hamiltonian.
    """

    # Put your solution here #

    # Define a device using qml.device
    dev = qml.device('default.qubit', wires=num_wires)

    @qml.qnode(dev)
    def circuit(num_wires):
        """
        A quantum circuit with Hadamard gates on every qubit and that measures
        the expectation value of the Hamiltonian in question. 
        
        Args:
        	num_wires (int): The number of qubits.

		Returns:
			(float): The expectation value of the Hamiltonian.
        """

        # Put Hadamard gates here #
        for i in range(num_wires):
            qml.Hadamard(wires=i)

        # Then return the expectation value of the Hamiltonian using qml.expval
        return qml.expval(hamiltonian(num_wires))

    return circuit(num_wires)

# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    num_wires = json.loads(test_case_input)
    output = expectation_value(num_wires)

    return str(output)


def check(solution_output: str, expected_output: str) -> None:
    print(f"Solution output: {solution_output}")
    print(f"Expected output: {expected_output}")
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(solution_output, expected_output, rtol=1e-4)

# These are the public test cases
test_cases = [
    ('8', '9.33333'),
    ('3', '1.00000')
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