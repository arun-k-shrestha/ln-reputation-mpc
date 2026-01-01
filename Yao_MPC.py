# Yao's Millionaires' Protocol Implementation
#source: https://www.youtube.com/watch?v=gf4BawfCY1A

# Example usage
import random
import math

# Shared transformation used by both parties
def sharedFunction(x):
    return (x * 7919 + 42) % 982451653  # Simple linear transformation

# Encodes a number by dividing with the shared randomNum1
def Inverse(x, r):
    return x / r

# Reverses the encoded number by multiplying with randomNum1
def reverseInverse(x, r):
    return r * x

# Simulated Yao's Millionaires' Protocol 
# Highest should always set to value greater than the value of Sender
def Yao_Millionaires_Protocol(Intermediate_Node, Sender, Highest, randomNum, randomNum1=None):
    if randomNum1 is None:
        randomNum1 = random.randint(1, 500)
    # Mask the sender's value using an inverse transformation
    masked_value = Inverse(randomNum, randomNum1) - Sender
    encoded_values = []

    # Generate hidden values from 0 to Highest
    # Highest should be the maximum possible balance in the network
    for i in range(0, Highest):
        hidden = reverseInverse(masked_value + i, randomNum1)
        transform = sharedFunction(hidden)
        encoded_values.append(transform)

    # Add +1 to values starting from Intermediate_Node to Highest
    lowerBound = round(Intermediate_Node)
    for i in range(lowerBound, Highest):
        encoded_values[i] += 1

    # Compute the comparison value (check if it's among the encoded values)
    checker = sharedFunction(randomNum) + 1

    #return any(val == checker for val in encoded_values)

    # Use safe float comparison
    if any(math.isclose(val, checker, rel_tol=1e-9) for val in encoded_values): # because when x = 0.1 + 0.2 the print(x == 0.3) -> False but print(math.isclose(x, 0.3)) ->  True
        return True  # Sender is richer than Intermediate_Node
    else:
        return False  # Sender is not richer