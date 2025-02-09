import json
import pennylane as qml
import pennylane.numpy as np
# Step 1: initialize a device
num_wires = 1
dev = qml.device("default.qubit", wires = num_wires) # Put your code here #

# Step 2: Add a decorator below
@qml.qnode(dev)
def simple_circuit(angle):

    """
    In this function:
        * Rotate the qubit around the y-axis by angle
        * Measure the expectation value of the Pauli X observable

    Args:
        angle (float): how much to rotate a state around the y-axis

    Returns:
        Union[tensor, float]: The expectation value of the Pauli X observable
    """
    

    # Step 3: Add gates to the QNode
    
    # Put your code here #
    qml.RY(angle, wires = 0)
    # Step 4: Return the required expectation value 
    return qml.probs(0)
# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    angle = json.loads(test_case_input)
    output = simple_circuit(angle)[0]

    return str(output)


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(solution_output, expected_output, rtol=1e-4)

test_cases = [
    ('1.45783', '0.5563631060725739'),
    ('0.9572', '0.7879057356348377')
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