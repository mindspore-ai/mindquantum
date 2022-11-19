#   Copyright 2022 <Huawei Technologies Co., Ltd>
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Base class to define arithmetic operators for some classes encapsulating a C++ object."""

import numbers

# pylint: disable=attribute-defined-outside-init


class CppArithmeticAdaptor:
    """
    Adaptor class to handle a Python class with a C++ underlying object.

    All arithmetic operations are simply forwarded to the underlying C++ object.

    Attributes:
        _cpp_obj (object): The underlying C++ object to which we should forward all operations
    """

    @property
    def is_complex(self):
        """Return whether the current instance is complex valued."""
        return NotImplemented

    @staticmethod
    def _valid_other(other):  # pylint: disable=unused-argument
        return NotImplemented

    # ----------------------------------

    def __len__(self):
        """Return length of CppArithmeticAdaptor."""
        return len(self._cpp_obj)

    # ----------------------------------

    def __neg__(self):
        """Return negative CppArithmeticAdaptor."""
        return self.__class__(-self._cpp_obj)

    # ----------------------------------

    def __add__(self, other):
        """Add a number or a CppArithmeticAdaptor."""
        if not self.__class__._valid_other(other):
            return NotImplemented
        if hasattr(other, '_cpp_obj'):
            return self.__class__(self._cpp_obj + other._cpp_obj)
        return self.__class__(self._cpp_obj + other)

    def __iadd__(self, other):
        """Inplace add a number or a CppArithmeticAdaptor."""
        if not self.is_complex and other.is_complex:
            self._cpp_obj = self._cpp_obj.cast_complex()

        if hasattr(other, '_cpp_obj'):
            self._cpp_obj += other._cpp_obj
        else:
            self._cpp_obj += other
        return self

    def __radd__(self, other):
        """Right add a number or a CppArithmeticAdaptor."""
        if not self.__class__._valid_other(other):
            return NotImplemented
        if hasattr(other, '_cpp_obj'):
            return self.__class__(self._cpp_obj + other._cpp_obj)
        return self.__class__(self._cpp_obj + other)

    # ----------------------------------

    def __sub__(self, other):
        """Subtract a number or a CppArithmeticAdaptor."""
        if not self.__class__._valid_other(other):
            return NotImplemented
        if hasattr(other, '_cpp_obj'):
            return self.__class__(self._cpp_obj - other._cpp_obj)
        return self.__class__(self._cpp_obj - other)

    def __isub__(self, other):
        """Inplace subtraction a number or a CppArithmeticAdaptor."""
        if not self.is_complex and other.is_complex:
            self._cpp_obj = self._cpp_obj.cast_complex()

        if hasattr(other, '_cpp_obj'):
            self._cpp_obj -= other._cpp_obj
        else:
            self._cpp_obj -= other

        return self

    def __rsub__(self, other):
        """Subtract a number or a CppArithmeticAdaptor with this CppArithmeticAdaptor."""
        if not self.__class__._valid_other(other):
            return NotImplemented

        if hasattr(other, '_cpp_obj'):
            return self.__class__(other._cpp_obj - self._cpp_obj)
        return self.__class__(other - self._cpp_obj)

    # ----------------------------------

    def __mul__(self, other):
        """Multiply by a number or a CppArithmeticAdaptor."""
        if not self.__class__._valid_other(other):
            return NotImplemented
        if hasattr(other, '_cpp_obj'):
            return self.__class__(self._cpp_obj * other._cpp_obj)
        return self.__class__(self._cpp_obj * other)

    def __imul__(self, other):
        """Inplace multiply by a number or a CppArithmeticAdaptor."""
        if not self.is_complex:
            if isinstance(other, numbers.Complex):
                if not isinstance(other, numbers.Real):
                    self._cpp_obj = self._cpp_obj.cast_complex()
            elif hasattr(other, 'is_complex') and other.is_complex:
                self._cpp_obj = self._cpp_obj.cast_complex()

        if hasattr(other, '_cpp_obj'):
            self._cpp_obj *= other._cpp_obj
        else:
            self._cpp_obj *= other
        return self

    def __rmul__(self, other):
        """Right multiply by a number or a CppArithmeticAdaptor."""
        if not self.__class__._valid_other(other):
            return NotImplemented

        if hasattr(other, '_cpp_obj'):
            return self.__class__(self._cpp_obj * other._cpp_obj)
        return self.__class__(self._cpp_obj * other)

    # ----------------------------------

    def __truediv__(self, other):
        """Divide a number."""
        if not self.__class__._valid_other(other):
            return NotImplemented
        if hasattr(other, '_cpp_obj'):
            return self.__class__(self._cpp_obj * other._cpp_obj)
        return self.__class__(self._cpp_obj / other)

    def __itruediv__(self, other):
        """Divide by a number or a CppArithmeticAdaptor."""
        if not self.is_complex:
            if isinstance(other, numbers.Complex):
                if not isinstance(other, numbers.Real):
                    self._cpp_obj = self._cpp_obj.cast_complex()
            elif hasattr(other, 'is_complex') and other.is_complex:
                self._cpp_obj = self._cpp_obj.cast_complex()

        if hasattr(other, '_cpp_obj'):
            self._cpp_obj /= other._cpp_obj
        else:
            self._cpp_obj /= other
        return self

    # ----------------------------------

    def __pow__(self, exponent: int):
        """Exponential of CppArithmeticAdaptors."""
        return self.__class__(self._cpp_obj**exponent)

    # ----------------------------------

    def __eq__(self, other) -> bool:
        """Check whether two CppArithmeticAdaptors equal."""
        if not self.is_complex:
            if isinstance(other, numbers.Complex) and not isinstance(other, numbers.Real):
                return self._cpp_obj.cast_complex() == other._cpp_obj
            if hasattr(other, "is_complex"):
                if other.is_complex:
                    return self._cpp_obj.cast_complex() == other._cpp_obj
        return self._cpp_obj == other._cpp_obj

    def __ne__(self, other) -> bool:
        """Check whether two CppArithmeticAdaptors not equal."""
        return not self._cpp_obj == other._cpp_obj
