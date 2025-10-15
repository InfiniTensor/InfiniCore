"""
Flexible parameter mapping system that allows operators to define their own mapping rules
"""

from .base import TestCase, TensorSpec


class ParameterMapping:
    """Base class for parameter mapping configurations"""

    def __init__(self, operator_name, call_signature, input_rules, output_rules):
        """
        Args:
            operator_name: Name of the operator
            call_signature: Function signature template, e.g., "matmul(a, b)" or "add(input, other)"
            input_rules: List of rules for mapping test case data to input specifications
            output_rules: Rules for mapping test case data to output specification
        """
        self.operator_name = operator_name
        self.call_signature = call_signature
        self.input_rules = input_rules
        self.output_rules = output_rules

    def map_test_case(self, test_case_data, operation_mode=TestCase.OUT_OF_PLACE):
        """Map test case data to TestCase object using defined rules"""
        # Normalize test case data to handle different input formats
        normalized_data = self._normalize_test_case_data(test_case_data)

        inputs = []

        # Process input rules
        for rule in self.input_rules:
            input_spec = self._apply_rule(rule, normalized_data)
            if input_spec is not None:
                inputs.append(input_spec)

        # Process output rules
        output_spec = self._apply_rule(self.output_rules, normalized_data)

        return TestCase(operation_mode, inputs=inputs, output=output_spec)

    def _normalize_test_case_data(self, test_case_data):
        """Normalize test case data to handle different input formats"""
        if not isinstance(test_case_data, (list, tuple)):
            return (test_case_data,)

        # If the first element is a tuple (shape), and there's only one element
        # e.g., ((13, 4)) → this should be shape (13, 4)
        if len(test_case_data) == 1 and isinstance(test_case_data[0], (list, tuple)):
            shape = test_case_data[0]
            return (shape, None, None, None)
        # If it's a tuple of integers (single shape), e.g., (13, 4)
        elif all(isinstance(x, int) for x in test_case_data):
            return (test_case_data, None, None, None)
        else:
            return test_case_data

    def _apply_rule(self, rule, test_case_data):
        """Apply a single mapping rule to test case data"""
        if rule is None:
            return None

        if callable(rule):
            # Rule is a function
            return rule(test_case_data)
        elif isinstance(rule, dict):
            # Rule is a dictionary with shape and stride specifications
            shape_rule = rule.get("shape")
            stride_rule = rule.get("stride")

            if shape_rule is None:
                return None

            # Get shape from test case data
            if callable(shape_rule):
                shape = shape_rule(test_case_data)
            else:
                shape = self._get_data_by_index(test_case_data, shape_rule)

            if shape is None:
                return None

            # Get stride from test case data
            stride = None
            if stride_rule is not None:
                if callable(stride_rule):
                    stride = stride_rule(test_case_data)
                else:
                    stride = self._get_data_by_index(test_case_data, stride_rule)

            # Only create strided tensor if stride is provided and valid
            if self._is_valid_strides(stride):
                return TensorSpec.from_strided_tensor(shape, stride)
            else:
                return TensorSpec.from_tensor(shape)
        else:
            raise ValueError(f"Invalid rule format: {rule}")

    def _get_data_by_index(self, test_case_data, index):
        """Safely get data from test case by index, return None if index out of range or value is None"""
        if isinstance(index, int):
            if 0 <= index < len(test_case_data):
                value = test_case_data[index]
                # Return None if the value is explicitly None
                return value if value is not None else None
            else:
                return None
        else:
            # If index is not an integer, assume it's a fixed value
            return index

    def _is_valid_strides(self, strides):
        """Check if strides are valid (not None and have proper format)"""
        if strides is None:
            return False
        if isinstance(strides, (list, tuple)) and len(strides) > 0:
            # Check if all elements are integers (or can be converted to valid strides)
            return all(isinstance(s, int) for s in strides)
        return False


def create_parameter_mapping(
    operator_name, call_signature, input_configs, output_config
):
    """
    Create a parameter mapping from configuration

    Args:
        operator_name: Name of the operator
        call_signature: Function call signature
        input_configs: List of input configurations
        output_config: Output configuration

    Example for matmul:
        input_configs = [
            {'shape': 0, 'stride': 3},  # a: shape from index 0, stride from index 3
            {'shape': 1, 'stride': 4}   # b: shape from index 1, stride from index 4
        ]
        output_config = {'shape': 2, 'stride': 5}  # output: shape from index 2, stride from index 5

    Example for add:
        input_configs = [
            {'shape': 0, 'stride': 1},  # input: shape from index 0, stride from index 1
            {'shape': 0, 'stride': 2}   # other: shape from index 0, stride from index 2
        ]
        output_config = {'shape': 0, 'stride': 3}  # output: shape from index 0, stride from index 3
    """
    return ParameterMapping(operator_name, call_signature, input_configs, output_config)


def create_test_cases(test_case_data, parameter_mapping):
    """
    Create test cases from data using specified parameter mapping

    Args:
        test_case_data: List of test case specifications with operation mode as first element
        parameter_mapping: ParameterMapping instance or configuration tuple

    Returns:
        List of TestCase objects
    """
    if isinstance(parameter_mapping, (list, tuple)):
        # Unpack configuration: (operator_name, call_signature, input_configs, output_config)
        if len(parameter_mapping) == 4:
            operator_name, call_signature, input_configs, output_config = (
                parameter_mapping
            )
            parameter_mapping = create_parameter_mapping(
                operator_name, call_signature, input_configs, output_config
            )
        else:
            raise ValueError("Invalid parameter mapping configuration format")

    test_cases = []
    for i, data in enumerate(test_case_data):
        if isinstance(data, TestCase):
            test_cases.append(data)
        else:
            try:
                # Extract operation mode from first element
                operation_mode = TestCase.OUT_OF_PLACE  # Default
                mapping_data = data

                if isinstance(data, (list, tuple)) and len(data) > 0:
                    if data[0] in [
                        TestCase.IN_PLACE,
                        TestCase.OUT_OF_PLACE,
                        TestCase.BOTH,
                    ]:
                        operation_mode = data[0]
                        mapping_data = data[1:]
                    else:
                        # Default to out-of-place if not specified
                        operation_mode = TestCase.OUT_OF_PLACE
                        mapping_data = data

                test_case = parameter_mapping.map_test_case(
                    mapping_data, operation_mode
                )
                test_cases.append(test_case)
            except Exception as e:
                print(f"Warning: Failed to map test case {i} data {data}: {e}")
                # Fallback: try to create TestCase directly
                if isinstance(data, (list, tuple)):
                    # Handle operation mode in fallback
                    if data[0] in [
                        TestCase.IN_PLACE,
                        TestCase.OUT_OF_PLACE,
                        TestCase.BOTH,
                    ]:
                        operation_mode = data[0]
                        test_data = data[1:]
                    else:
                        operation_mode = TestCase.OUT_OF_PLACE
                        test_data = data

                    # For fallback, assume first element is inputs, second is output
                    if len(test_data) >= 1:
                        inputs = (
                            test_data[0]
                            if isinstance(test_data[0], (list, tuple))
                            else [test_data[0]]
                        )
                        output = test_data[1] if len(test_data) > 1 else None
                        test_cases.append(TestCase(operation_mode, inputs, output))
                    else:
                        print(
                            f"Warning: Skipping test case {i} due to insufficient data: {data}"
                        )
                else:
                    test_cases.append(TestCase(TestCase.OUT_OF_PLACE, [data]))

    return test_cases
