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
#
# Copyright (c) 2013-2015 SymPy Development Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# pylint: disable=useless-suppression

"""SymEngine compatibility header."""

import re as _re
import string
from itertools import product as cartes

try:
    from ._mindquantum_cxx import symengine

    _range = _re.compile('([0-9]*:[0-9]+|[a-zA-Z]?:[a-zA-Z])')

    def symbols(names, **args):  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        r"""
        Transform strings into instances of :class:`Symbol` class.

        :func:`symbols` function returns a sequence of symbols with names taken
        from ``names`` argument, which can be a comma or whitespace delimited
        string, or a sequence of strings::
            >>> from mindquantum.experimental.symengine import symbols
            >>> x, y, z = symbols('x,y,z')
            >>> a, b, c = symbols('a b c')
        The type of output is dependent on the properties of input arguments::
            >>> symbols('x')
            x
            >>> symbols('x,')
            (x,)
            >>> symbols('x,y')
            (x, y)
            >>> symbols(('a', 'b', 'c'))
            (a, b, c)
            >>> symbols(['a', 'b', 'c'])
            [a, b, c]
            >>> symbols(set(['a', 'b', 'c']))
            set([a, b, c])
        If an iterable container is needed for a single symbol, set the ``seq``
        argument to ``True`` or terminate the symbol name with a comma::
            >>> symbols('x', seq=True)
            (x,)
        To reduce typing, range syntax is supported to create indexed symbols.
        Ranges are indicated by a colon and the type of range is determined by
        the character to the right of the colon. If the character is a digit
        then all contiguous digits to the left are taken as the nonnegative
        starting value (or 0 if there is no digit left of the colon) and all
        contiguous digits to the right are taken as 1 greater than the ending
        value::
            >>> symbols('x:10')
            (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9)
            >>> symbols('x5:10')
            (x5, x6, x7, x8, x9)
            >>> symbols('x5(:2)')
            (x50, x51)
            >>> symbols('x5:10,y:5')
            (x5, x6, x7, x8, x9, y0, y1, y2, y3, y4)
            >>> symbols(('x5:10', 'y:5'))
            ((x5, x6, x7, x8, x9), (y0, y1, y2, y3, y4))
        If the character to the right of the colon is a letter, then the single
        letter to the left (or 'a' if there is none) is taken as the start
        and all characters in the lexicographic range *through* the letter to
        the right are used as the range::
            >>> symbols('x:z')
            (x, y, z)
            >>> symbols('x:c')  # null range
            ()
            >>> symbols('x(:c)')
            (xa, xb, xc)
            >>> symbols(':c')
            (a, b, c)
            >>> symbols('a:d, x:z')
            (a, b, c, d, x, y, z)
            >>> symbols(('a:d', 'x:z'))
            ((a, b, c, d), (x, y, z))
        Multiple ranges are supported; contiguous numerical ranges should be
        separated by parentheses to disambiguate the ending number of one
        range from the starting number of the next::
            >>> symbols('x:2(1:3)')
            (x01, x02, x11, x12)
            >>> symbols(':3:2')  # parsing is from left to right
            (00, 01, 10, 11, 20, 21)
        Only one pair of parentheses surrounding ranges are removed, so to
        include parentheses around ranges, double them. And to include spaces,
        commas, or colons, escape them with a backslash::
            >>> symbols('x((a:b))')
            (x(a), x(b))
            >>> symbols('x(:1\,:2)')  # or 'x((:1)\,(:2))'
            (x(0,0), x(0,1))
        """
        result = []

        if isinstance(names, str):
            marker = 0
            literals = [r'\,', r'\:', r'\ ']
            for i in range(len(literals)):
                lit = literals.pop(0)
                if lit in names:
                    while chr(marker) in names:
                        marker += 1
                    lit_char = chr(marker)
                    marker += 1
                    names = names.replace(lit, lit_char)
                    literals.append((lit_char, lit[1:]))

            def literal(lit):
                if literals:
                    for char, literal in literals:
                        lit = lit.replace(char, literal)
                return lit

            names = names.strip()
            as_seq = names.endswith(',')
            if as_seq:
                names = names[:-1].rstrip()
            if not names:
                raise ValueError('no symbols given')

            # split on commas
            names = [n.strip() for n in names.split(',')]
            if not all(n for n in names):
                raise ValueError('missing symbol between commas')
            # split on spaces
            for i in range(len(names) - 1, -1, -1):
                names[i : i + 1] = names[i].split()  # noqa: E203

            cls = args.pop('cls', symengine.symbol)  # pylint: disable=no-member
            seq = args.pop('seq', as_seq)

            for name in names:
                if not name:
                    raise ValueError('missing symbol')

                if ':' not in name:
                    result.append(cls(literal(name), **args))
                    continue

                split = _range.split(name)
                # remove 1 layer of bounding parentheses around ranges
                for i in range(len(split) - 1):
                    if (
                        i
                        and ':' in split[i]
                        and split[i] != ':'
                        and split[i - 1].endswith('(')
                        and split[i + 1].startswith(')')
                    ):
                        split[i - 1] = split[i - 1][:-1]
                        split[i + 1] = split[i + 1][1:]
                for i, symbol in enumerate(split):
                    if ':' in symbol:
                        if symbol[-1].endswith(':'):
                            raise ValueError('missing end range')
                        left_symbol, right_symbol = symbol.split(':')
                        if right_symbol[-1] in string.digits:
                            left_symbol = 0 if not left_symbol else int(left_symbol)
                            right_symbol = int(right_symbol)
                            split[i] = [str(c) for c in range(left_symbol, right_symbol)]
                        else:
                            left_symbol = left_symbol or 'a'
                            split[i] = [
                                string.ascii_letters[c]
                                for c in range(
                                    string.ascii_letters.index(left_symbol),
                                    string.ascii_letters.index(right_symbol) + 1,
                                )
                            ]  # inclusive
                        if not split[i]:
                            break
                    else:
                        split[i] = [symbol]
                else:
                    seq = True
                    if len(split) == 1:
                        names = split[0]
                    else:
                        names = [''.join(s) for s in cartes(*split)]
                    if literals:
                        result.extend([cls(literal(s), **args) for s in names])
                    else:
                        result.extend([cls(s, **args) for s in names])

            if not seq and len(result) <= 1:
                if not result:
                    return ()
                return result[0]

            return tuple(result)

        # else
        for name in names:
            result.append(symbols(name, **args))
        return type(names)(result)

    def var(names, **args):
        """
        Create symbols and inject them into the global namespace.

        INPUT:
        -            s -- a string, either a single variable name, or
        -                 a space separated list of variable names, or
        -                 a list of variable names.

        This calls :func:`symbols` with the same arguments and puts the results
        into the *global* namespace. It's recommended not to use :func:`var` in
        library code, where :func:`symbols` has to be used::

        Examples
        ========

        >>> from symengine import var

        >>> var('x')
        x
        >>> x
        x

        >>> var('a,ab,abc')
        (a, ab, abc)
        >>> abc
        abc

        See :func:`symbols` documentation for more details on what kinds of
        arguments can be passed to :func:`var`.
        """

        def traverse(symbols_arg, frame):
            """Recursively inject symbols to the global namespace."""
            for symbol in symbols_arg:
                if isinstance(symbol, symengine.Basic):  # pylint: disable=no-member
                    frame.f_globals[str(symbol)] = symbol
                # Once we have an undefined function class implemented, put a check for function here
                else:
                    traverse(symbol, frame)

        from inspect import currentframe  # pylint: disable=import-outside-toplevel

        frame = currentframe().f_back

        try:
            syms = symbols(names, **args)

            if syms is not None:
                if isinstance(syms, symengine.Basic):  # pylint: disable=no-member
                    frame.f_globals[str(syms)] = syms
                # Once we have an undefined function class implemented, put a check for function here
                else:
                    traverse(syms, frame)
        finally:
            del frame  # break cyclic dependencies as stated in inspect docs

        return syms

except ImportError:

    def symbols(names, **args):  # pylint: disable=unused-argument
        """Transform strings into instances of :class:`Symbol` class."""
        raise RuntimeError('Experimental C++ module not compiled!')
