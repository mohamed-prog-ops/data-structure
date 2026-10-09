"""
Stack-Based Mathematical Expression Evaluator
=============================================
Student Name: Mohamed E Amara
Student ID:   905005067
Class:        DIT1202F
Institution:  Limkokwing University of Creative Technology, Sierra Leone
Module:       Data Structures and Algorithm (DCOMP320)
Assignment:   Stacks Implementation
Examiner:     Amandus Benjamin Coker

Description:
This program implements a custom Stack data structure from scratch using
Python lists. It evaluates arithmetic expressions read from an input file
using the Shunting-Yard algorithm (infix-to-postfix conversion) followed by
stack-based postfix evaluation, and outputs formatted results to an output file.
"""

import os
import sys
from typing import List, Union


class Stack:
    """
    A Last-In, First-Out (LIFO) Stack implementation using a Python list internally.
    """

    def __init__(self):
        """Initialize an empty stack."""
        self._items: List[Union[int, float, str]] = []

    def push(self, item: Union[int, float, str]) -> None:
        """
        Push an item onto the top of the stack.

        :param item: The element to add to the stack.
        """
        self._items.append(item)

    def pop(self) -> Union[int, float, str]:
        """
        Remove and return the top item from the stack.

        :return: The top element of the stack.
        :raises IndexError: If the stack is empty.
        """
        if self.is_empty():
            raise IndexError("Cannot pop from an empty stack.")
        return self._items.pop()

    def peek(self) -> Union[int, float, str]:
        """
        Return the top item from the stack without removing it.

        :return: The top element of the stack.
        :raises IndexError: If the stack is empty.
        """
        if self.is_empty():
            raise IndexError("Cannot peek into an empty stack.")
        return self._items[-1]

    def is_empty(self) -> bool:
        """
        Check if the stack is empty.

        :return: True if the stack has no items, False otherwise.
        """
        return len(self._items) == 0

    def size(self) -> int:
        """
        Return the number of items in the stack.

        :return: Integer representing stack size.
        """
        return len(self._items)

    def __len__(self) -> int:
        """Support len() operator on Stack."""
        return len(self._items)

    def __repr__(self) -> str:
        """Return string representation of the stack for debugging."""
        return f"Stack({self._items})"


# Operator definitions: precedence and associativity ('L' for Left, 'R' for Right)
OPERATORS = {
    '+':  {'precedence': 1, 'assoc': 'L'},
    '-':  {'precedence': 1, 'assoc': 'L'},
    '*':  {'precedence': 2, 'assoc': 'L'},
    '/':  {'precedence': 2, 'assoc': 'L'},
    'u+': {'precedence': 3, 'assoc': 'R'},  # Unary plus
    'u-': {'precedence': 3, 'assoc': 'R'},  # Unary minus (negation)
}


def tokenize(expression: str) -> List[str]:
    """
    Break an arithmetic expression string into individual tokens.

    Supports multi-digit numbers, decimal numbers, arithmetic operators (+, -, *, /),
    unary signs (+, -), and parentheses.

    :param expression: The raw arithmetic string.
    :return: List of string tokens.
    :raises ValueError: If an invalid character is encountered.
    """
    tokens: List[str] = []
    i = 0
    n = len(expression)

    while i < n:
        char = expression[i]

        # Ignore whitespace
        if char.isspace():
            i += 1
            continue

        # Handle numeric values (integers and decimals)
        if char.isdigit() or (char == '.' and i + 1 < n and expression[i + 1].isdigit()):
            num_str = []
            has_dot = False
            while i < n and (expression[i].isdigit() or expression[i] == '.'):
                if expression[i] == '.':
                    if has_dot:
                        raise ValueError("Invalid number format: multiple decimal points.")
                    has_dot = True
                num_str.append(expression[i])
                i += 1
            tokens.append(''.join(num_str))
            continue

        # Handle parentheses
        if char in ('(', ')'):
            tokens.append(char)
            i += 1
            continue

        # Handle operators and distinguish binary vs. unary operators
        if char in ('+', '-', '*', '/'):
            # A '+' or '-' is unary if:
            # 1. It is at the beginning of the expression, or
            # 2. It immediately follows another operator or an opening parenthesis '('
            if char in ('+', '-'):
                prev_token = tokens[-1] if tokens else None
                is_unary = (
                    prev_token is None
                    or prev_token in ('+', '-', '*', '/', 'u+', 'u-', '(')
                )
                if is_unary:
                    tokens.append('u+' if char == '+' else 'u-')
                    i += 1
                    continue

            tokens.append(char)
            i += 1
            continue

        # Any other character is invalid
        raise ValueError(f"Invalid character encountered: '{char}'")

    return tokens


def infix_to_postfix(tokens: List[str]) -> List[str]:
    """
    Convert an infix list of tokens to postfix (Reverse Polish Notation)
    using Dijkstra's Shunting-Yard Algorithm with a custom Stack.

    :param tokens: List of infix tokens.
    :return: List of postfix tokens.
    :raises ValueError: On mismatched parentheses or syntax errors.
    """
    output: List[str] = []
    operator_stack = Stack()

    for token in tokens:
        # Check if token is a number
        if token not in OPERATORS and token not in ('(', ')'):
            output.append(token)

        elif token in OPERATORS:
            curr_prec = OPERATORS[token]['precedence']
            curr_assoc = OPERATORS[token]['assoc']

            while not operator_stack.is_empty():
                top = operator_stack.peek()
                if top in OPERATORS:
                    top_prec = OPERATORS[top]['precedence']
                    if (curr_assoc == 'L' and curr_prec <= top_prec) or \
                       (curr_assoc == 'R' and curr_prec < top_prec):
                        output.append(operator_stack.pop())
                    else:
                        break
                else:
                    break

            operator_stack.push(token)

        elif token == '(':
            operator_stack.push(token)

        elif token == ')':
            found_opening = False
            while not operator_stack.is_empty():
                top = operator_stack.pop()
                if top == '(':
                    found_opening = True
                    break
                output.append(top)

            if not found_opening:
                raise ValueError("Mismatched parentheses: missing opening parenthesis '('.")

    # Pop any remaining operators from the stack
    while not operator_stack.is_empty():
        top = operator_stack.pop()
        if top in ('(', ')'):
            raise ValueError("Mismatched parentheses: missing closing parenthesis ')'.")
        output.append(top)

    return output


def evaluate_postfix(postfix_tokens: List[str]) -> float:
    """
    Evaluate a postfix expression using a custom Stack.

    :param postfix_tokens: List of postfix tokens.
    :return: The evaluated numerical result as a float.
    :raises ZeroDivisionError: If division by zero occurs.
    :raises ValueError: If expression is malformed or operands are insufficient.
    """
    if not postfix_tokens:
        raise ValueError("Empty expression.")

    eval_stack = Stack()

    for token in postfix_tokens:
        if token in OPERATORS:
            if token == 'u+':
                # Unary plus: value remains unchanged
                if eval_stack.is_empty():
                    raise ValueError("Insufficient operands for unary '+'.")
                continue
            elif token == 'u-':
                # Unary minus (negation)
                if eval_stack.is_empty():
                    raise ValueError("Insufficient operands for unary '-'.")
                val = eval_stack.pop()
                eval_stack.push(-float(val))
            else:
                # Binary operator
                if eval_stack.size() < 2:
                    raise ValueError(f"Insufficient operands for operator '{token}'.")
                b = float(eval_stack.pop())
                a = float(eval_stack.pop())

                if token == '+':
                    eval_stack.push(a + b)
                elif token == '-':
                    eval_stack.push(a - b)
                elif token == '*':
                    eval_stack.push(a * b)
                elif token == '/':
                    if b == 0:
                        raise ZeroDivisionError("Division by zero.")
                    eval_stack.push(a / b)
        else:
            try:
                num = float(token)
                eval_stack.push(num)
            except ValueError:
                raise ValueError(f"Invalid numeric value: '{token}'.")

    if eval_stack.size() != 1:
        raise ValueError("Malformed expression: excess operands or operators.")

    return float(eval_stack.pop())


def format_result(value: float) -> str:
    """
    Format the calculated float result according to the project specifications:
    - Whole-number results formatted as integers (e.g. 2 instead of 2.0).
    - Non-whole results formatted to a maximum of 4 decimal places with unnecessary trailing zeros stripped.
    - Negative zero (-0, -0.0) normalized to 0.

    :param value: Calculated numeric value.
    :return: Formatted string representation.
    """
    # Normalize potential negative zero or near-zero float artifacts
    if abs(value) < 1e-12:
        return "0"

    # Check if the value is an exact integer
    if value.is_integer():
        return str(int(value))

    # Format with up to 4 decimal places
    formatted = f"{value:.4f}".rstrip('0').rstrip('.')

    # Guard against "-0" resulting from rounding
    if formatted in ("-0", "-0.0", "-"):
        return "0"

    return formatted


def is_separator_line(line: str) -> bool:
    """
    Determine if a line is a separator line.
    A stripped line containing two or more identical characters from '-=_*~#'
    (e.g., '---', '===', '###', '***', '~~~', '___', '--', '##') is treated as a separator.

    :param line: Raw text line.
    :return: True if the line is a separator line, False otherwise.
    """
    s = line.strip()
    return len(s) >= 2 and len(set(s)) == 1 and s[0] in "-=_*~#"


def evaluate_expression(expr_str: str) -> str:
    """
    Tokenize, convert, and evaluate a single arithmetic expression string.

    :param expr_str: The expression to evaluate.
    :return: Formatted result string or an 'Error: ...' message.
    """
    try:
        tokens = tokenize(expr_str)
        if not tokens:
            return ""
        postfix = infix_to_postfix(tokens)
        result = evaluate_postfix(postfix)
        return format_result(result)
    except ZeroDivisionError as zde:
        return f"Error: {zde}"
    except Exception as e:
        return f"Error: {e}"


def process_file(input_filename: str = "input.txt", output_filename: str = "output.txt") -> bool:
    """
    Read expressions from the input file, evaluate each, and write results to the output file.

    Preserves blank lines and separator lines.
    If the input file does not exist, prints an error and exits cleanly without creating output.txt.

    :param input_filename: Path to input text file.
    :param output_filename: Path to output text file.
    :return: True if processing succeeded, False if input file was missing.
    """
    if not os.path.exists(input_filename):
        print(f"Error: Input file '{input_filename}' not found.")
        return False

    with open(input_filename, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    output_lines: List[str] = []

    for line in lines:
        raw_line = line.rstrip("\r\n")

        # Preserve empty / blank lines
        if raw_line.strip() == "":
            output_lines.append("")
            continue

        # Preserve separator lines exactly as they appear
        if is_separator_line(raw_line):
            output_lines.append(raw_line)
            continue

        # Process mathematical expression
        res = evaluate_expression(raw_line)
        output_lines.append(res)

    with open(output_filename, "w", encoding="utf-8") as outfile:
        outfile.write("\n".join(output_lines) + "\n")

    print(f"Successfully processed '{input_filename}' -> '{output_filename}'.")
    return True


if __name__ == "__main__":
    # Default file processing execution
    in_file = sys.argv[1] if len(sys.argv) > 1 else "input.txt"
    out_file = sys.argv[2] if len(sys.argv) > 2 else "output.txt"
    success = process_file(in_file, out_file)
    if not success:
        sys.exit(1)
