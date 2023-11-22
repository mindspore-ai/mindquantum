"""utility functions."""

import math
import numpy as np


def _fill_fist_sep(string, sep, length, fill_char=' '):
    poi = string.find(sep)
    if length < poi:
        raise Exception(
            f"Original length is {poi}, can not fill it to length {length}.")
    return string[:poi] + fill_char * (length - poi) + string[poi:]


def _check_str(string, name):
    if not isinstance(string, str):
        raise TypeError(f"{name} requires str, but get {type(string)}!")


# pylint: disable=too-many-arguments
def bprint(strings: list, align=":", title='', v_around='=', h_around='|', fill_char=' '):
    """
    Print the information in block shape.
    Refer: https://gitee.com/mindspore/mindquantum

    Args:
        strings (list[str]): Information you want to output.
        align (str): The align char alone vertal direction. Default: ``":"``.
        title (str): The title of this information block. Default: ``""``.
        v_around (str): Vertical boundary char. Default: ``"="``.
        h_around (str): horizontal boundary char. Default: ``"|"``.
        fill_char (str): Empty space fill with this char. Default: ``" "``.

    Returns:
        list, a list of formatted string.

    Examples:
        >>> from mindquantum.io import bprint
        >>> title='Info of Bob'
        >>> o = bprint(['Name:Bob', 'Age:17', 'Nationality:China'],
        ...     title=title)
        >>> for i in o:
        ...     print(i)
        ====Info of Bob====
        |Name       :Bob  |
        |Age        :17   |
        |Nationality:China|
        ===================
    """
    if not isinstance(strings, list):
        raise TypeError(f"strings requires a list, but get {type(strings)}")
    for string in strings:
        _check_str(string, "string")
    _check_str(align, 'align')
    _check_str(title, 'title')
    _check_str(v_around, 'v_around')
    _check_str(h_around, 'h_around')
    _check_str(fill_char, 'fill_char')
    maxmim_len = strings[0].find(align)
    for sub_str in strings:
        m_poi = sub_str.find(align)
        if m_poi > maxmim_len:
            maxmim_len = m_poi
    strings = [_fill_fist_sep(i, align, maxmim_len, fill_char)
               for i in strings]
    n_around = 3
    title = v_around * n_around + title + v_around * n_around
    maxmim = max(len(i) for i in strings)
    if len(title) > maxmim:
        len_total = (len(title) - maxmim) // 2 + (len(title) - maxmim) % 2
        strings = [h_around + ' ' * len_total + i + ' ' *
                   (len(title) - len(i) - len_total) + h_around for i in strings]
        title = h_around + title + h_around
    else:
        len_total = (maxmim - len(title)) // 2 + (maxmim - len(title)) % 2
        title = v_around + v_around * len_total + \
            title + v_around * len_total + v_around
        strings = [h_around + i + ' ' *
                   (len(title) - 2 - len(i)) + h_around for i in strings]
    bot = v_around + v_around * (len(title) - 2) + v_around
    output = []
    output.append(title)
    output.extend(strings)
    output.append(bot)
    return output


def str_special(str_pr):
    """Represent the string in more concise way.
    Refer: https://github.com/GhostArtyom/QuditVQE/tree/main/QuditSim
    """
    special = {'': 1, 'π': np.pi, '√2': np.sqrt(
        2), '√3': np.sqrt(3), '√5': np.sqrt(5)}
    if isinstance(str_pr, (int, str)):
        return str(str_pr)
    elif str_pr % 1 == 0:
        return str(int(str_pr))
    div = -1 if str_pr < 0 else 1
    str_pr *= -1 if str_pr < 0 else 1
    for key, val in special.items():
        if isinstance(str_pr, str):
            break
        if np.isclose(str_pr / val % 1, 0):
            div *= int(str_pr / val)
            str_pr = key if div == 1 else f'-{key}' if div == - \
                1 else f'{div}{key}'
        elif np.isclose(val / str_pr % 1, 0):
            div *= int(val / str_pr)
            key = 1 if val == 1 else key
            str_pr = f'{key}/{div}' if div > 0 else f'-{key}/{-div}'
    if isinstance(str_pr, str):
        return str_pr
    return str(round(str_pr * div, 4))


def str_ket(dim: int, state: np.ndarray) -> str:
    """Get ket format of the qudit state.
    Refer: https://github.com/GhostArtyom/QuditVQE/tree/main/QuditSim
    """
    if state.ndim == 2 and (state.shape[0] == 1 or state.shape[1] == 1):
        state = state.flatten()
    if state.ndim != 1:
        raise ValueError(
            f'State requires a 1-D ndarray, but get {state.shape}')
    nq = round(math.log(len(state), dim), 12)
    if nq % 1 != 0:
        raise ValueError(
            f'Wrong state shape {state.shape} is not a power of {dim}')
    nq = int(nq)
    tol = 1e-8
    string = []
    for ind, val in enumerate(state):
        base = np.base_repr(ind, dim).zfill(nq)
        real = np.real(val)
        imag = np.imag(val)
        str_real = str_special(real)
        str_imag = str_special(imag)
        if np.abs(val) < tol:
            continue
        if np.abs(real) < tol:
            string.append(f'{str_imag}j¦{base}⟩')
            continue
        if np.abs(imag) < tol:
            string.append(f'{str_real}¦{base}⟩')
            continue
        if str_imag.startswith('-'):
            string.append(f'{str_real}{str_imag}j¦{base}⟩')
        else:
            string.append(f'{str_real}+{str_imag}j¦{base}⟩')
    return '\n'.join(string)


__all__ = ['bprint', 'str_special', 'str_ket']
