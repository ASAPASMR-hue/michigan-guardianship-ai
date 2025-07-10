def add_two_numbers(a, b):
    """
    Adds two numbers together and returns the result.
    
    Args:
        a (int or float): First number
        b (int or float): Second number
        
    Returns:
        int or float: Sum of a and b
    """
    return a + b


if __name__ == "__main__":
    # Example usage
    result = add_two_numbers(5, 3)
    print(f"5 + 3 = {result}")
    
    result = add_two_numbers(2.5, 1.5)
    print(f"2.5 + 1.5 = {result}")
    
    result = add_two_numbers(-10, 15)
    print(f"-10 + 15 = {result}")