# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# wITHOUT wARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Test svg."""

import pytest

from mindquantum import RX, BarrierGate, Simulator, qft


def test_measure_svg():
    """
    Description: Test measure result svg.
    Expectation: success.
    """
    circ = qft(range(3)).measure_all()
    sim = Simulator('projectq', 3)
    res = sim.sampling(circ, shots=100, seed=42)
    text = res.svg()._repr_svg_().split('bar')
    text = "bar".join([text[0]] + ['"'.join(i.split('"')[1:]) for i in text[1:]])
    text_exp = (
        '<div class="nb-html-output output_area"><svg xmlns="http://www.w3.org/2000/svg" width="415.6" height="327.0" '
        'xmlns:xlink="http://www.w3.org/1999/xlink">\n<rect x="0" y="0" width="415.6" height="327.0" fill="#ffffff" '
        '/>\n<text x="10" y="17.0" font-size="14px" dominant-baseline="middle" text-anchor="start" >\nShots:\n 100\n '
        '</text>\n<text x="10" y="31.0" font-size="14px" dominant-baseline="middle" text-anchor="start" >\nKeys: q0 q1 '
        'q2\n </text>\n<line x1="45.599999999999994" x2="405.6" y1="62.0" y2="62.0" stroke="#adb0b8" stroke-width="1" '
        '/>\n<line x1="45.599999999999994" x2="45.599999999999994" y1="55.0" y2="62.0" stroke="#adb0b8" '
        'stroke-width="1" />\n<text x="47.599999999999994" y="60.0" font-size="12px" dominant-baseline="bottom" '
        'text-anchor="start" fill="#575d6c" >\n0.0\n </text>\n<line x1="45.599999999999994" x2="45.599999999999994" '
        'y1="62.0" y2="317.0" stroke="#adb0b8" stroke-width="1" />\n<line x1="105.6" x2="105.6" y1="55.0" y2="62.0" '
        'stroke="#adb0b8" stroke-width="1" />\n<text x="107.6" y="60.0" font-size="12px" dominant-baseline="bottom" '
        'text-anchor="start" fill="#575d6c" >\n0.032\n </text>\n<line x1="105.6" x2="105.6" y1="62.0" y2="317.0" '
        'stroke="#dfe1e6" stroke-width="1" />\n<line x1="165.6" x2="165.6" y1="55.0" y2="62.0" stroke="#adb0b8" '
        'stroke-width="1" />\n<text x="167.6" y="60.0" font-size="12px" dominant-baseline="bottom" text-anchor="start" '
        'fill="#575d6c" >\n0.064\n </text>\n<line x1="165.6" x2="165.6" y1="62.0" y2="317.0" stroke="#dfe1e6" '
        'stroke-width="1" />\n<line x1="225.6" x2="225.6" y1="55.0" y2="62.0" stroke="#adb0b8" stroke-width="1" '
        '/>\n<text x="227.6" y="60.0" font-size="12px" dominant-baseline="bottom" text-anchor="start" fill="#575d6c" '
        '>\n0.096\n </text>\n<line x1="225.6" x2="225.6" y1="62.0" y2="317.0" stroke="#dfe1e6" stroke-width="1" '
        '/>\n<line x1="285.6" x2="285.6" y1="55.0" y2="62.0" stroke="#adb0b8" stroke-width="1" />\n<text x="287.6" '
        'y="60.0" font-size="12px" dominant-baseline="bottom" text-anchor="start" fill="#575d6c" >\n0.128\n '
        '</text>\n<line x1="285.6" x2="285.6" y1="62.0" y2="317.0" stroke="#dfe1e6" stroke-width="1" />\n<line '
        'x1="345.6" x2="345.6" y1="55.0" y2="62.0" stroke="#adb0b8" stroke-width="1" />\n<text x="347.6" y="60.0" '
        'font-size="12px" dominant-baseline="bottom" text-anchor="start" fill="#575d6c" >\n0.16\n </text>\n<line '
        'x1="345.6" x2="345.6" y1="62.0" y2="317.0" stroke="#dfe1e6" stroke-width="1" />\n<text x="36.599999999999994" '
        'y="85.0" font-size="12px" dominant-baseline="middle" text-anchor="end" fill="#575d6c" >\n000\n </text>\n<line '
        'x1="38.599999999999994" x2="45.599999999999994" y1="85.0" y2="85.0" stroke="#adb0b8" stroke-width="1" '
        '/>\n<rect x="45.599999999999994" y="73.0" width="262.5" height="24" id="bar fill="#5e7ce0" />\n<text '
        'x="318.1" y="85.0" font-size="14px" dominant-baseline="middle" text-anchor="start" fill="#575d6c" id="bar '
        'fill-opacity="0" >\n14\n </text>\n<text x="36.599999999999994" y="115.0" font-size="12px" '
        'dominant-baseline="middle" text-anchor="end" fill="#575d6c" >\n001\n </text>\n<line x1="38.599999999999994" '
        'x2="45.599999999999994" y1="115.0" y2="115.0" stroke="#adb0b8" stroke-width="1" />\n<rect '
        'x="45.599999999999994" y="103.0" width="262.5" height="24" id="bar fill="#16acff" />\n<text x="318.1" '
        'y="115.0" font-size="14px" dominant-baseline="middle" text-anchor="start" fill="#575d6c" id="bar '
        'fill-opacity="0" >\n14\n </text>\n<text x="36.599999999999994" y="145.0" font-size="12px" '
        'dominant-baseline="middle" text-anchor="end" fill="#575d6c" >\n010\n </text>\n<line x1="38.599999999999994" '
        'x2="45.599999999999994" y1="145.0" y2="145.0" stroke="#adb0b8" stroke-width="1" />\n<rect '
        'x="45.599999999999994" y="133.0" width="243.75" height="24" id="bar fill="#5e7ce0" />\n<text x="299.35" '
        'y="145.0" font-size="14px" dominant-baseline="middle" text-anchor="start" fill="#575d6c" id="bar '
        'fill-opacity="0" >\n13\n </text>\n<text x="36.599999999999994" y="175.0" font-size="12px" '
        'dominant-baseline="middle" text-anchor="end" fill="#575d6c" >\n011\n </text>\n<line x1="38.599999999999994" '
        'x2="45.599999999999994" y1="175.0" y2="175.0" stroke="#adb0b8" stroke-width="1" />\n<rect '
        'x="45.599999999999994" y="163.0" width="300.0" height="24" id="bar fill="#16acff" />\n<text x="355.6" '
        'y="175.0" font-size="14px" dominant-baseline="middle" text-anchor="start" fill="#575d6c" id="bar '
        'fill-opacity="0" >\n16\n </text>\n<text x="36.599999999999994" y="205.0" font-size="12px" '
        'dominant-baseline="middle" text-anchor="end" fill="#575d6c" >\n100\n </text>\n<line x1="38.599999999999994" '
        'x2="45.599999999999994" y1="205.0" y2="205.0" stroke="#adb0b8" stroke-width="1" />\n<rect '
        'x="45.599999999999994" y="193.0" width="206.25" height="24" id="bar fill="#5e7ce0" />\n<text x="261.85" '
        'y="205.0" font-size="14px" dominant-baseline="middle" text-anchor="start" fill="#575d6c" id="bar '
        'fill-opacity="0" >\n11\n </text>\n<text x="36.599999999999994" y="235.0" font-size="12px" '
        'dominant-baseline="middle" text-anchor="end" fill="#575d6c" >\n101\n </text>\n<line x1="38.599999999999994" '
        'x2="45.599999999999994" y1="235.0" y2="235.0" stroke="#adb0b8" stroke-width="1" />\n<rect '
        'x="45.599999999999994" y="223.0" width="112.5" height="24" id="bar fill="#16acff" />\n<text x="168.1" '
        'y="235.0" font-size="14px" dominant-baseline="middle" text-anchor="start" fill="#575d6c" id="bar '
        'fill-opacity="0" >\n6\n </text>\n<text x="36.599999999999994" y="265.0" font-size="12px" '
        'dominant-baseline="middle" text-anchor="end" fill="#575d6c" >\n110\n </text>\n<line x1="38.599999999999994" '
        'x2="45.599999999999994" y1="265.0" y2="265.0" stroke="#adb0b8" stroke-width="1" />\n<rect '
        'x="45.599999999999994" y="253.0" width="243.75" height="24" id="bar fill="#5e7ce0" />\n<text x="299.35" '
        'y="265.0" font-size="14px" dominant-baseline="middle" text-anchor="start" fill="#575d6c" id="bar '
        'fill-opacity="0" >\n13\n </text>\n<text x="36.599999999999994" y="295.0" font-size="12px" '
        'dominant-baseline="middle" text-anchor="end" fill="#575d6c" >\n111\n </text>\n<line x1="38.599999999999994" '
        'x2="45.599999999999994" y1="295.0" y2="295.0" stroke="#adb0b8" stroke-width="1" />\n<rect '
        'x="45.599999999999994" y="283.0" width="243.75" height="24" id="bar fill="#16acff" />\n<text x="299.35" '
        'y="295.0" font-size="14px" dominant-baseline="middle" text-anchor="start" fill="#575d6c" id="bar '
        'fill-opacity="0" >\n13\n </text>\n<animate xlink:href="#bar attributeName="width" from="0" to="262.5" '
        'dur="0.3s" calcMode="spline" values="0; 262.5" keyTimes="0; 1" keySplines="0.42 0 1 0.8;" fill="freeze" '
        '/>\n<animate xlink:href="#bar attributeName="width" from="0" to="262.5" dur="0.3s" calcMode="spline" '
        'values="0; 262.5" keyTimes="0; 1" keySplines="0.42 0 1 0.8;" fill="freeze" />\n<animate xlink:href="#bar '
        'attributeName="width" from="0" to="243.75" dur="0.3s" calcMode="spline" values="0; 243.75" keyTimes="0; 1" '
        'keySplines="0.42 0 1 0.8;" fill="freeze" />\n<animate xlink:href="#bar attributeName="width" from="0" '
        'to="300.0" dur="0.3s" calcMode="spline" values="0; 300.0" keyTimes="0; 1" keySplines="0.42 0 1 0.8;" '
        'fill="freeze" />\n<animate xlink:href="#bar attributeName="width" from="0" to="206.25" dur="0.3s" '
        'calcMode="spline" values="0; 206.25" keyTimes="0; 1" keySplines="0.42 0 1 0.8;" fill="freeze" />\n<animate '
        'xlink:href="#bar attributeName="width" from="0" to="112.5" dur="0.3s" calcMode="spline" values="0; 112.5" '
        'keyTimes="0; 1" keySplines="0.42 0 1 0.8;" fill="freeze" />\n<animate xlink:href="#bar attributeName="width" '
        'from="0" to="243.75" dur="0.3s" calcMode="spline" values="0; 243.75" keyTimes="0; 1" keySplines="0.42 0 1 '
        '0.8;" fill="freeze" />\n<animate xlink:href="#bar attributeName="width" from="0" to="243.75" dur="0.3s" '
        'calcMode="spline" values="0; 243.75" keyTimes="0; 1" keySplines="0.42 0 1 0.8;" fill="freeze" />\n<animate '
        'xlink:href="#bar attributeName="fill" from="#16acff" to="#fac209" dur="0.15s" calcMode="spline" '
        'values="#16acff; #fac209" keyTimes="0; 1" keySplines="0.42 0 1 0.8;" fill="freeze" begin="0.3s" />\n<animate '
        'xlink:href="#bar attributeName="fill-opacity" from="0" to="1" dur="0.15s" calcMode="spline" values="0; 1" '
        'keyTimes="0; 1" keySplines="0.42 0 1 0.8;" fill="freeze" begin="0.3s" />\n<animate xlink:href="#bar '
        'attributeName="fill-opacity" from="0" to="1" dur="0.15s" calcMode="spline" values="0; 1" keyTimes="0; 1" '
        'keySplines="0.42 0 1 0.8;" fill="freeze" begin="0.3s" />\n<animate xlink:href="#bar '
        'attributeName="fill-opacity" from="0" to="1" dur="0.15s" calcMode="spline" values="0; 1" keyTimes="0; 1" '
        'keySplines="0.42 0 1 0.8;" fill="freeze" begin="0.3s" />\n<animate xlink:href="#bar '
        'attributeName="fill-opacity" from="0" to="1" dur="0.15s" calcMode="spline" values="0; 1" keyTimes="0; 1" '
        'keySplines="0.42 0 1 0.8;" fill="freeze" begin="0.3s" />\n<animate xlink:href="#bar '
        'attributeName="fill-opacity" from="0" to="1" dur="0.15s" calcMode="spline" values="0; 1" keyTimes="0; 1" '
        'keySplines="0.42 0 1 0.8;" fill="freeze" begin="0.3s" />\n<animate xlink:href="#bar '
        'attributeName="fill-opacity" from="0" to="1" dur="0.15s" calcMode="spline" values="0; 1" keyTimes="0; 1" '
        'keySplines="0.42 0 1 0.8;" fill="freeze" begin="0.3s" />\n<animate xlink:href="#bar '
        'attributeName="fill-opacity" from="0" to="1" dur="0.15s" calcMode="spline" values="0; 1" keyTimes="0; 1" '
        'keySplines="0.42 0 1 0.8;" fill="freeze" begin="0.3s" />\n<animate xlink:href="#bar '
        'attributeName="fill-opacity" from="0" to="1" dur="0.15s" calcMode="spline" values="0; 1" keyTimes="0; 1" '
        'keySplines="0.42 0 1 0.8;" fill="freeze" begin="0.3s" />\n<text x="210.3" y="41.0" font-size="14px" '
        'dominant-baseline="middle" text-anchor="middle" >\nprobability\n </text>\n</svg></div>'
    )
    assert text == text_exp


@pytest.mark.xfail
def test_circuit_svg():
    """
    Description: Test svg default style.
    Expectation: success.
    """
    text = (qft(range(3)) + RX({'a': 1.2}).on(1) + BarrierGate()).measure_all().svg()._repr_svg_()
    text_exp = (
        '<div class="nb-html-output output_area"><svg xmlns="http://www.w3.org/2000/svg" width="716.8" height="200"'
        'xmlns:xlink="http://www.w3.org/1999/xlink">\n<rect x="0" y="0" width="716.8" height="200" fill="#ffffff"'
        '/>\n<text x="20.0" y="40.0" font-size="16px" dominant-baseline="middle" text-anchor="start"'
        'font-family="Arial" font-weight="normal" fill="#252b3a" >\nq0:\n </text>\n<text x="20.0" y="100.0"'
        'font-size="16px" dominant-baseline="middle" text-anchor="start" font-family="Arial" font-weight="normal"'
        'fill="#252b3a" >\nq1:\n </text>\n<text x="20.0" y="160.0" font-size="16px" dominant-baseline="middle"'
        'text-anchor="start" font-family="Arial" font-weight="normal" fill="#252b3a" >\nq2:\n </text>\n<line x1="48.8"'
        'x2="696.8" y1="40.0" y2="40.0" stroke="#adb0b8" stroke-width="1" />\n<line x1="48.8" x2="696.8" y1="100.0"'
        'y2="100.0" stroke="#adb0b8" stroke-width="1" />\n<line x1="48.8" x2="696.8" y1="160.0" y2="160.0"'
        'stroke="#adb0b8" stroke-width="1" />\n\n<rect x="72.8" y="20.0" width="40.0" height="40" rx="4" ry="4"'
        'stroke="#ffffff" stroke-width="0" fill="#5e7ce0" fill-opacity="1" />\n<text x="92.8" y="40.0" font-size="20px"'
        'dominant-baseline="middle" text-anchor="middle" font-family="Arial" font-weight="normal" fill="#ffffff" >\nH\n'
        '</text>\n\n<circle cx="152.8" cy="100.0" r="4" fill="#fac209" />\n<line x1="152.8" x2="152.8" y1="40.0"'
        'y2="100.0" stroke="#fac209" stroke-width="3" />\n<rect x="132.8" y="20.0" width="40.0" height="40" rx="4"'
        'ry="4" stroke="#ffffff" stroke-width="0" fill="#fac209" fill-opacity="1" />\n<text x="152.8" y="36.0"'
        'font-size="20px" dominant-baseline="middle" text-anchor="middle" font-family="Arial" font-weight="normal"'
        'fill="#ffffff" >\nPS\n </text>\n<text x="152.8" y="52.0" font-size="14.0px" dominant-baseline="middle"'
        'text-anchor="middle" font-family="Arial" font-weight="normal" fill="#ffffff" >\nπ/2\n </text>\n\n<circle'
        'cx="212.8" cy="160.0" r="4" fill="#fac209" />\n<line x1="212.8" x2="212.8" y1="40.0" y2="160.0"'
        'stroke="#fac209" stroke-width="3" />\n<rect x="192.8" y="20.0" width="40.0" height="40" rx="4" ry="4"'
        'stroke="#ffffff" stroke-width="0" fill="#fac209" fill-opacity="1" />\n<text x="212.8" y="36.0"'
        'font-size="20px" dominant-baseline="middle" text-anchor="middle" font-family="Arial" font-weight="normal"'
        'fill="#ffffff" >\nPS\n </text>\n<text x="212.8" y="52.0" font-size="14.0px" dominant-baseline="middle"'
        'text-anchor="middle" font-family="Arial" font-weight="normal" fill="#ffffff" >\nπ/4\n </text>\n\n\n<rect'
        'x="252.8" y="80.0" width="40.0" height="40" rx="4" ry="4" stroke="#ffffff" stroke-width="0" fill="#5e7ce0"'
        'fill-opacity="1" />\n<text x="272.8" y="100.0" font-size="20px" dominant-baseline="middle"'
        'text-anchor="middle" font-family="Arial" font-weight="normal" fill="#ffffff" >\nH\n </text>\n\n<circle'
        'cx="332.8" cy="160.0" r="4" fill="#fac209" />\n<line x1="332.8" x2="332.8" y1="100.0" y2="160.0"'
        'stroke="#fac209" stroke-width="3" />\n<rect x="312.8" y="80.0" width="40.0" height="40" rx="4" ry="4"'
        'stroke="#ffffff" stroke-width="0" fill="#fac209" fill-opacity="1" />\n<text x="332.8" y="96.0"'
        'font-size="20px" dominant-baseline="middle" text-anchor="middle" font-family="Arial" font-weight="normal"'
        'fill="#ffffff" >\nPS\n </text>\n<text x="332.8" y="112.0" font-size="14.0px" dominant-baseline="middle"'
        'text-anchor="middle" font-family="Arial" font-weight="normal" fill="#ffffff" >\nπ/2\n </text>\n\n\n<rect'
        'x="372.8" y="140.0" width="40.0" height="40" rx="4" ry="4" stroke="#ffffff" stroke-width="0" fill="#5e7ce0"'
        'fill-opacity="1" />\n<text x="392.8" y="160.0" font-size="20px" dominant-baseline="middle"'
        'text-anchor="middle" font-family="Arial" font-weight="normal" fill="#ffffff" >\nH\n </text>\n\n<line'
        'x1="452.8" x2="452.8" y1="20.0" y2="180.0" stroke-width="3" stroke="#16acff" />\n\n<rect x="432.8" y="20.0"'
        'width="40" height="40" rx="4" ry="4" fill="#16acff" fill-opacity="1" stroke="#ffffff" stroke-width="0"'
        '/>\n<path d="M 443.2 36.31384387633061 L 448.0 28.0 L 452.8 36.31384387633061 L 449.44 36.31384387633061 L'
        '449.44 52.0 L 446.56 52.0 L 446.56 36.31384387633061 Z" fill="#ffffff" />\n<path d="M 462.40000000000003'
        '43.68615612366939 L 457.6 52.0 L 452.8 43.68615612366939 L 456.16 43.68615612366939 L 456.16 28.0 L 459.04'
        '28.0 L 459.04 43.68615612366939 Z" fill="#ffffff" />\n<rect x="432.8" y="140.0" width="40" height="40" rx="4"'
        'ry="4" fill="#16acff" fill-opacity="1" stroke="#ffffff" stroke-width="0" />\n<path d="M 443.2'
        '156.31384387633062 L 448.0 148.0 L 452.8 156.31384387633062 L 449.44 156.31384387633062 L 449.44 172.0 L'
        '446.56 172.0 L 446.56 156.31384387633062 Z" fill="#ffffff" />\n<path d="M 462.40000000000003'
        '163.68615612366938 L 457.6 172.0 L 452.8 163.68615612366938 L 456.16 163.68615612366938 L 456.16 148.0 L'
        '459.04 148.0 L 459.04 163.68615612366938 Z" fill="#ffffff" />\n\n<rect x="492.8" y="80.0" width="80.0"'
        'height="40" rx="4" ry="4" stroke="#ffffff" stroke-width="0" fill="#fac209" fill-opacity="1" />\n<text'
        'x="532.8" y="96.0" font-size="20px" dominant-baseline="middle" text-anchor="middle" font-family="Arial"'
        'font-weight="normal" fill="#ffffff" >\nRX\n </text>\n<text x="532.8" y="112.0" font-size="14.0px"'
        'dominant-baseline="middle" text-anchor="middle" font-family="Arial" font-weight="normal" fill="#ffffff"'
        '>\n6/5*a\n </text>\n\n<rect x="592.8" y="20.0" width="20" height="160" fill="gray" fill-opacity="0.8"'
        '/>\n<rect x="632.8" y="20.0" width="40" height="40" rx="4" ry="4" stroke="#ffffff" stroke-width="0"'
        'fill="#ff7272" fill-opacity="1" />\n<circle cx="652.8" cy="50.4" r="1.6" fill="#ffffff" />\n<path d="M 640.0'
        '50.4 A 12.8 12.8 0 0 1 665.5999999999999 50.4" stroke="#ffffff" stroke-width="2.4000000000000004"'
        'fill-opacity="0" />\n<path d="M 656.9018483174859 33.93539030917347 L 665.2156921938165 29.135390309173467 L'
        '665.2156921938165 38.73539030917347 L 661.8901546432843 36.815390309173466 L 654.0470765814496 50.4 L'
        '652.3843078061834 49.44 L 660.2273858680181 35.85539030917347 Z" fill="#ffffff" />\n<rect x="632.8" y="80.0"'
        'width="40" height="40" rx="4" ry="4" stroke="#ffffff" stroke-width="0" fill="#ff7272" fill-opacity="1"'
        '/>\n<circle cx="652.8" cy="110.4" r="1.6" fill="#ffffff" />\n<path d="M 640.0 110.4 A 12.8 12.8 0 0 1'
        '665.5999999999999 110.4" stroke="#ffffff" stroke-width="2.4000000000000004" fill-opacity="0" />\n<path d="M'
        '656.9018483174859 93.93539030917347 L 665.2156921938165 89.13539030917347 L 665.2156921938165'
        '98.73539030917347 L 661.8901546432843 96.81539030917347 L 654.0470765814496 110.4 L 652.3843078061834 109.44 L'
        '660.2273858680181 95.85539030917347 Z" fill="#ffffff" />\n<rect x="632.8" y="140.0" width="40" height="40"'
        'rx="4" ry="4" stroke="#ffffff" stroke-width="0" fill="#ff7272" fill-opacity="1" />\n<circle cx="652.8"'
        'cy="170.4" r="1.6" fill="#ffffff" />\n<path d="M 640.0 170.4 A 12.8 12.8 0 0 1 665.5999999999999 170.4"'
        'stroke="#ffffff" stroke-width="2.4000000000000004" fill-opacity="0" />\n<path d="M 656.9018483174859'
        '153.93539030917347 L 665.2156921938165 149.13539030917346 L 665.2156921938165 158.73539030917345 L'
        '661.8901546432843 156.81539030917347 L 654.0470765814496 170.4 L 652.3843078061834 169.44 L 660.2273858680181'
        '155.85539030917346 Z" fill="#ffffff" />\n</svg></div>'
    )
    assert text == text_exp
