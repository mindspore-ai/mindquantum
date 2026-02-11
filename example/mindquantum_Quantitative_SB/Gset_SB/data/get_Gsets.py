import requests
import os


def download_gsets(output_dir='./', start=1, end=21):
    """
    从 https://web.stanford.edu/~yyye/yyye/Gset/ 下载Gsets数据(G1-G21)到同目录下
    
    参数:
        output_dir: 数据保存目录 (默认当前脚本所在目录)
        start: 起始Gset编号 (默认1)
        end: 结束Gset编号 (默认21)
    """
    base_url = 'https://web.stanford.edu/~yyye/yyye/Gset/'
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_output_dir = os.path.join(script_dir, output_dir) if output_dir else script_dir
    
    # 确保输出目录存在
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)
    
    for i in range(start, end + 1):
        filename = f'G{i}.txt'
        url = f'{base_url}/G{i}'
        
        try:
            print(f'正在下载 {filename}...')
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            filepath = os.path.join(full_output_dir, filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f'{filename} 下载成功')
        except requests.exceptions.RequestException as e:
            print(f'{filename} 下载失败: {e}')


def download_optimal_solutions(output_dir='./', gset_list=None):
    """
    下载Gset数据的精确解文件

    参数:
        output_dir: 数据保存目录 (默认当前脚本所在目录)
        gset_list: 需要下载精确解的Gset编号列表 (默认[1, 11, 13, 32, 48])
    """
    base_url = 'https://web.stanford.edu/~yyye/yyye/Gset'

    if gset_list is None:
        gset_list = [1, 11, 13, 32, 48]

    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_output_dir = os.path.join(script_dir, output_dir) if output_dir else script_dir

    # 确保输出目录存在
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)

    print('\n开始下载精确解文件...')
    for gset_num in gset_list:
        url_filename = f'maxG{gset_num}'
        save_filename = f'maxG{gset_num}.txt'
        url = f'{base_url}/{url_filename}'

        try:
            print(f'正在下载 {save_filename}...')
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            filepath = os.path.join(full_output_dir, save_filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f'{save_filename} 下载成功')
        except requests.exceptions.RequestException as e:
            print(f'{save_filename} 下载失败: {e}')


if __name__ == '__main__':
    # 下载G1-G21数据到脚本同目录
    download_gsets(start=21, end=22)
    
    # 下载精确解文件 (G1, G11, G13, G32, G48)
    # download_optimal_solutions(gset_list=[1, 11, 13, 32, 48])
