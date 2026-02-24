"""
A very basic hello-world program that invokes a compute function and returns
a string along with result of the greet compute function.
"""

import fc


@fc.compute
def greet(arg: str):
    return f"Greetings, {arg}"


@fc.flow
def hello_world(arg: str):
    ret1 = greet(arg="Hi, Guest1")

    return (
        f"hello_world invoked with arg: {arg}. "
        f"Greet compute returned: {ret1} "
    )
