import json
import pennylane as qml
import pennylane.numpy as np
# Step 1: initialize a device by the name dev
dev = qml.device('default.qubit', wires = 2)
# Step 2: Add a decorator below
@qml.qnode(dev)
def simple_circuit(angle):

    """
    In this function:
        * Prepare the Bell state |Phi+>.
        * Rotate the first qubit around the y-axis by angle
        * Measure the tensor product observable Z0xZ1.

    Args:
        angle (float): how much to rotate a state around the y-axis.

    Returns:
        Union[tensor, float]: the expectation value of the Z0xZ1 observable.
    """
    

    # Step 3: Add gates to the QNode
    
    # Put your code here #
    qml.Hadamard(wires = 0)
    qml.CNOT(wires = [0, 1])
    qml.RY(angle, wires = 0)
    # Step 4: Return the required expectation value  
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    angle = json.loads(test_case_input)
    output = simple_circuit(angle)

    return str(output)

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(solution_output, expected_output, rtol=1e-4), "Not the right expectation value"

# These are the public test cases
test_cases = [
    ('1.23456', '0.3299365180851774'),
    ('1.86923', '-0.2940234756205866')
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