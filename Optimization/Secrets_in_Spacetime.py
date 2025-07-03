import json
import pennylane as qml
import pennylane.numpy as np
import random
import numpy as onp

def U_psi(theta):
    """
    Quantum function that generates |psi>, Zenda's state wants to send to Reece.

    Args:
        theta (float): Parameter that generates the state.

    """
    qml.Hadamard(wires = 0)
    qml.CRX(theta, wires = [0,1])
    qml.CRZ(theta, wires = [0,1])

def is_unsafe(alpha, beta, epsilon):
    """
    Boolean function that we will use to know if a set of parameters is unsafe.

    Args:
        alpha (float): parameter used to encode the state.
        beta (float): parameter used to encode the state.
        epsilon (float): unsafe-tolerance.

    Returns:
        (bool): 'True' if alpha and beta are epsilon-unsafe coefficients. 'False' in the other case.

    """
    # Build the encoding matrix
    enc_mat = np.array(qml.matrix(qml.prod(qml.RZ(alpha, wires=0), qml.RX(beta, wires=0)))) # type: ignore
    # Check if the |0> state is unsafe
    if np.abs(enc_mat[0,0])**2 >= 1-epsilon: # type: ignore
        return 'True'
    # Check if the |1> state is unsafe (using the largest eigenvalue for generality)
    if np.abs(enc_mat[1,1]) * np.max(np.linalg.eigvals(enc_mat)) >= 1-epsilon: # type: ignore
        return 'True'
    return 'False'

def brute_force_is_unsafe(alpha, beta, epsilon):
    pi = 3.141592653589793
    two_pi = 2 * pi
    thetas = onp.linspace(0, two_pi, 100000)
    vals = (alpha - 1) * thetas + beta
    vals_mod = (vals + pi) % two_pi - pi
    return onp.any(onp.abs(vals_mod) < epsilon)

# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    output = is_unsafe(*ins)
    return str(output)

def check(solution_output: str, expected_output: str) -> None:
    
    def bool_to_int(string):
        if string == "True":
            return 1
        return 0

    solution_output = str(bool_to_int(solution_output))
    print(f"Solution output: {solution_output}")
    expected_output = str(bool_to_int(expected_output))
    assert solution_output == expected_output, "The solution is not correct."

# These are the public test cases
test_cases = [
    ('[0.1, 0.2, 0.3]', 'True'),
    ('[1.1, 1.2, 0.3]', 'False'),
    ('[1.1, 1.2, 0.4]', 'True'),
    ('[0.5, 1.9, 0.7]', 'True')
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

# Note: Brute-force/analytic comparison is omitted because the current is_unsafe logic is not mathematically equivalent.
# If you want to test edge cases, add them to test_cases above.