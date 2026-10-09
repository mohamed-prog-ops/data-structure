"""
Unit Test Suite for Stack-Based Mathematical Expression Evaluator
=================================================================
Student Name: Mohamed E Amara
Student ID:   905005067
Class:        DIT1202F
Module:       Data Structures and Algorithm (DCOMP320)
"""

import os
import unittest
import tempfile
from stack_evaluator import (
    Stack,
    tokenize,
    infix_to_postfix,
    evaluate_postfix,
    format_result,
    is_separator_line,
    evaluate_expression,
    process_file
)


class TestStackDataStructure(unittest.TestCase):
    """Test custom Stack implementation for correct LIFO behavior and edge cases."""

    def setUp(self):
        self.stack = Stack()

    def test_empty_stack_initialization(self):
        self.assertTrue(self.stack.is_empty())
        self.assertEqual(self.stack.size(), 0)
        self.assertEqual(len(self.stack), 0)

    def test_push_and_peek(self):
        self.stack.push(42)
        self.assertFalse(self.stack.is_empty())
        self.assertEqual(self.stack.size(), 1)
        self.assertEqual(self.stack.peek(), 42)
        # Verify peek does not remove the item
        self.assertEqual(self.stack.size(), 1)

    def test_lifo_order(self):
        self.stack.push("first")
        self.stack.push("second")
        self.stack.push("third")

        self.assertEqual(self.stack.pop(), "third")
        self.assertEqual(self.stack.pop(), "second")
        self.assertEqual(self.stack.pop(), "first")
        self.assertTrue(self.stack.is_empty())

    def test_empty_stack_pop_raises_index_error(self):
        with self.assertRaises(IndexError):
            self.stack.pop()

    def test_empty_stack_peek_raises_index_error(self):
        with self.assertRaises(IndexError):
            self.stack.peek()


class TestArithmeticOperations(unittest.TestCase):
    """Test all four basic arithmetic operators (+, -, *, /)."""

    def test_addition(self):
        self.assertEqual(evaluate_expression("12 + 8"), "20")

    def test_subtraction(self):
        self.assertEqual(evaluate_expression("25 - 9"), "16")

    def test_multiplication(self):
        self.assertEqual(evaluate_expression("6 * 7"), "42")

    def test_division(self):
        self.assertEqual(evaluate_expression("84 / 2"), "42")


class TestPrecedenceAndAssociativity(unittest.TestCase):
    """Test operator precedence (*, / over +, -) and left-associativity."""

    def test_precedence_multiplication_over_addition(self):
        # 3 + (5 * 2) = 13, not (3 + 5) * 2 = 16
        self.assertEqual(evaluate_expression("3 + 5 * 2"), "13")

    def test_precedence_division_over_subtraction(self):
        # 20 - (10 / 2) = 15
        self.assertEqual(evaluate_expression("20 - 10 / 2"), "15")

    def test_left_associativity_subtraction(self):
        # (10 - 4) - 2 = 4, not 10 - (4 - 2) = 8
        self.assertEqual(evaluate_expression("10 - 4 - 2"), "4")

    def test_left_associativity_division(self):
        # (24 / 4) / 2 = 3, not 24 / (4 / 2) = 12
        self.assertEqual(evaluate_expression("24 / 4 / 2"), "3")

    def test_mixed_precedence_and_associativity(self):
        # 10 + 6 * 2 - 8 / 4 = 10 + 12 - 2 = 20
        self.assertEqual(evaluate_expression("10 + 6 * 2 - 8 / 4"), "20")


class TestParenthesesHandling(unittest.TestCase):
    """Test nested, simple, and mismatched parentheses."""

    def test_simple_parentheses(self):
        self.assertEqual(evaluate_expression("(3 + 5) * 2"), "16")

    def test_nested_parentheses(self):
        self.assertEqual(evaluate_expression("((2 + 3) * (4 + 1)) / 5"), "5")

    def test_deeply_nested_parentheses(self):
        self.assertEqual(evaluate_expression("(((10 - 2) * 3) + 6) / 6"), "5")

    def test_mismatched_missing_closing(self):
        res = evaluate_expression("(3 + 5 * 2")
        self.assertTrue(res.startswith("Error:"), f"Expected error but got '{res}'")
        self.assertIn("Mismatched parentheses", res)

    def test_mismatched_missing_opening(self):
        res = evaluate_expression("3 + 5) * 2")
        self.assertTrue(res.startswith("Error:"), f"Expected error but got '{res}'")
        self.assertIn("Mismatched parentheses", res)


class TestNumbersAndFormats(unittest.TestCase):
    """Test multi-digit numbers, decimal inputs, and formatting rules."""

    def test_multi_digit_numbers(self):
        self.assertEqual(evaluate_expression("1000 + 250 * 4"), "2000")

    def test_decimals_input(self):
        self.assertEqual(evaluate_expression("3.5 + 2.25 * 2"), "8")

    def test_fractional_results_formatted_to_4_decimals(self):
        # 1 / 3 = 0.333333... -> "0.3333"
        self.assertEqual(evaluate_expression("1 / 3"), "0.3333")
        # 7 / 8 = 0.875 (trailing zeros stripped)
        self.assertEqual(evaluate_expression("7 / 8"), "0.875")
        # 1 / 8 = 0.125
        self.assertEqual(evaluate_expression("1 / 8"), "0.125")

    def test_negative_results(self):
        self.assertEqual(evaluate_expression("5 - 15"), "-10")

    def test_negative_zero_normalization(self):
        # 0 * -5 or -0.0 should format as "0"
        self.assertEqual(format_result(-0.0), "0")
        self.assertEqual(evaluate_expression("0 * -1"), "0")


class TestWhitespaceVariations(unittest.TestCase):
    """Test expressions with various spacing styles."""

    def test_no_spaces(self):
        self.assertEqual(evaluate_expression("3+5*2"), "13")
        self.assertEqual(evaluate_expression("(8/4)+7*2"), "16")
        self.assertEqual(evaluate_expression("10-(2+3)*4"), "-10")

    def test_excessive_whitespace(self):
        self.assertEqual(evaluate_expression("   10   -   (  2  + 3 )  *  4  "), "-10")


class TestErrorHandling(unittest.TestCase):
    """Test division by zero, invalid characters, and malformed expressions."""

    def test_division_by_zero(self):
        res = evaluate_expression("10 / 0")
        self.assertTrue(res.startswith("Error:"))
        self.assertIn("Division by zero", res)

    def test_nested_division_by_zero(self):
        res = evaluate_expression("15 / (5 - 5)")
        self.assertTrue(res.startswith("Error:"))
        self.assertIn("Division by zero", res)

    def test_invalid_characters(self):
        res = evaluate_expression("3 + 5 * a")
        self.assertTrue(res.startswith("Error:"))
        self.assertIn("Invalid character", res)

        res2 = evaluate_expression("10 $ 2")
        self.assertTrue(res2.startswith("Error:"))

    def test_consecutive_operators_malformed(self):
        res = evaluate_expression("3 + * 2")
        self.assertTrue(res.startswith("Error:"))

    def test_trailing_operator(self):
        res = evaluate_expression("5 +")
        self.assertTrue(res.startswith("Error:"))


class TestSeparatorAndBlankLinePreservation(unittest.TestCase):
    """Test separator line recognition and blank line handling."""

    def test_separator_detection(self):
        self.assertTrue(is_separator_line("---"))
        self.assertTrue(is_separator_line("======"))
        self.assertTrue(is_separator_line("___"))
        self.assertTrue(is_separator_line("***"))
        self.assertTrue(is_separator_line("~~~"))
        self.assertTrue(is_separator_line("#####"))
        self.assertTrue(is_separator_line("##"))
        self.assertTrue(is_separator_line("--"))

        # Non-separators
        self.assertFalse(is_separator_line("-"))       # length < 2
        self.assertFalse(is_separator_line("-=-"))     # not identical chars
        self.assertFalse(is_separator_line("123"))     # not in allowed symbols
        self.assertFalse(is_separator_line(""))        # blank


class TestOfficialSampleAndFileProcessing(unittest.TestCase):
    """Test the complete official sample and file I/O operations."""

    def test_official_sample_evaluation(self):
        # Expression 1
        self.assertEqual(evaluate_expression("3 + 5 * 2"), "13")
        # Expression 2 (Corrected from brief's typo: (8/4) + 7*2 = 2 + 14 = 16)
        self.assertEqual(evaluate_expression("(8 / 4) + 7 * 2"), "16")
        # Expression 3
        self.assertEqual(evaluate_expression("10 - (2 + 3) * 4"), "-10")

    def test_official_sample_with_both_separators_end_to_end(self):
        sample_input = (
            "3 + 5 * 2\n"
            "---\n"
            "(8 / 4) + 7 * 2\n"
            "---\n"
            "10 - (2 + 3) * 4\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "official_input.txt")
            out_path = os.path.join(tmpdir, "official_output.txt")

            with open(in_path, "w", encoding="utf-8") as f:
                f.write(sample_input)

            success = process_file(in_path, out_path)
            self.assertTrue(success)
            self.assertTrue(os.path.exists(out_path))

            with open(out_path, "r", encoding="utf-8") as f:
                output_content = f.read()

            expected_lines = [
                "13",
                "---",
                "16",
                "---",
                "-10"
            ]
            self.assertEqual(output_content.strip().split("\n"), expected_lines)

    def test_file_processing_workflow(self):
        sample_input = (
            "3 + 5 * 2\n"
            "\n"
            "(8 / 4) + 7 * 2\n"
            "---\n"
            "10 - (2 + 3) * 4\n"
            "10 / 0\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "test_input.txt")
            out_path = os.path.join(tmpdir, "test_output.txt")

            with open(in_path, "w", encoding="utf-8") as f:
                f.write(sample_input)

            success = process_file(in_path, out_path)
            self.assertTrue(success)
            self.assertTrue(os.path.exists(out_path))

            with open(out_path, "r", encoding="utf-8") as f:
                output_content = f.read()

            expected_lines = [
                "13",
                "",
                "16",
                "---",
                "-10",
                "Error: Division by zero."
            ]
            self.assertEqual(output_content.strip().split("\n"), expected_lines)

    def test_missing_input_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            non_existent_in = os.path.join(tmpdir, "non_existent.txt")
            out_path = os.path.join(tmpdir, "should_not_exist.txt")

            success = process_file(non_existent_in, out_path)
            self.assertFalse(success)
            self.assertFalse(os.path.exists(out_path))


if __name__ == "__main__":
    unittest.main()
