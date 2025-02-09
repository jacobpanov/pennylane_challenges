import json
import pennylane as qml
import pennylane.numpy as np
def add(a: int, b: int) -> int:
    """Adds two integers.
    
    Args:
        a (int): One of the integers you should add together.
        b (int): One of the integers you should add together.
    Returns:
        (int): The sum of a+b of the given integers.
    """

    # Put your code here
    return a+b

# These functions are responsible for testing the solution.
def check(have: str, want: str) -> None:
    assert have == want


def run(case: str) -> str:
    answer = add(*map(int, case.split(" ")))
    return str(answer)

# These are the public test cases
test_cases = [
    ('1 1', '2'),
    ('2 3', '5')
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