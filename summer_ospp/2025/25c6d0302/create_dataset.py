import os
import time
import xmlrpc.client
import networkx as nx
import numpy as np
import email

# --- 交互参数 ---
MY_USERNAME = ""
MY_PASSWORD = ""
MY_EMAIL = ""
DATA_FILE = "test.dat"
POLLING_DELAY_SECOND = 15 # 每15秒查询一次状态
MAX_WAIT_MINUTES = 20 # 最长等待20分钟

# ---  NEOS 服务器交互函数 ---
def submit_to_neos() -> tuple:
    """
    通过API向NEOS服务器提交一个BiqMac任务。

    参数:
        biqmac_data_string (str): BiqMac格式的图数据。
        email_address (str): 用于接收通知的电子邮件地址（可选）。

    返回:
        tuple: (任务编号, 任务密码) 或 (None, None) 如果提交失败。
    """
    NEOS_URL = "https://neos-server.org:3333"
    # --- 从文件读取 XML Job Description ---
    with open(DATA_FILE, 'r') as f:
        data_content = f.read()
    try:
        neos = xmlrpc.client.ServerProxy(NEOS_URL)
        
        # 构建提交任务所需的XML字符串
        xml = f"""
                <document>
                <exampleName>BiqMac Example</exampleName>
                <category>co</category>
                <solver>BiqMac</solver>
                <inputMethod>SPARSE</inputMethod>
                <inputfile><![CDATA[ 
                {data_content}
                ]]></inputfile>
                <email>{MY_EMAIL}</email>
                <problem><![CDATA[mac]]></problem>
                </document>
                """
        
        job_number, password = neos.authenticatedSubmitJob(xml, MY_USERNAME, MY_PASSWORD)
        # job_number, password = neos.submitJob(xml)
        print(f"    成功提交任务到 NEOS. 任务编号: {job_number}")
        return job_number, password
    except Exception as e:
        print(f"    提交到 NEOS 失败: {e}")
        return None, None

def poll_and_get_results(job_number, password) -> str:
    """
    轮询NEOS服务器直到任务完成，并获取结果。

    参数:
        job_number (int): NEOS任务编号。
        password (str): NEOS任务密码。

    返回:
        str: 包含结果的输出日志字符串。
    """
    NEOS_URL = "https://neos-server.org:3333"
    neos = xmlrpc.client.ServerProxy(NEOS_URL)
    
    start_time = time.time()
    while time.time() - start_time < MAX_WAIT_MINUTES * 60:
        status = neos.getJobStatus(job_number, password)
        if status == "Done":
            print(f"    任务完成. 正在获取结果...")
            # NEOS返回的结果是经过MIME编码的字节流
            result_bytes = neos.getFinalResults(job_number, password).data
            # print(result_bytes)
            # 使用email库解析
            msg = email.message_from_bytes(result_bytes)
            print(msg)
            # 提取主要内容
            for part in msg.walk():
                if part.get_content_maintype() == 'text':
                    return part.get_payload(decode=True).decode('utf-8')
        
        print(f"    任务状态: {status}. 等待 {POLLING_DELAY_SECOND} 秒...")
        time.sleep(POLLING_DELAY_SECOND)
        
    print(f"    任务超时（超过 {MAX_WAIT_MINUTES} 分钟），已停止。")
    return None


def parse_neos_output_string(output_string: str) -> tuple:
    """
    从NEOS BiqMac的输出日志字符串中解析出最大割值和解向量。
    """
    max_cut_value = None
    solution_vector = None
    lines = output_string.splitlines()

    for i, line in enumerate(lines):
        if "cut value:" in line:
            try:
                max_cut_value = int(round(float(line.split()[-1])))
            except (ValueError, IndexError):
                continue
        
        if "one side of the cut:" in line and i + 1 < len(lines):
            try:
                solution_line = lines[i+1].strip()
                solution_vector = [int(x) for x in solution_line.split()]
            except (ValueError):
                continue
    
    if max_cut_value is not None and solution_vector is not None:
         print(f"    成功解析结果. 最大割值: {max_cut_value}")
    else:
         print("    未能从NEOS输出中解析出完整结果。")
         
    return max_cut_value, solution_vector


# --- 图生成和数据转换函数 ---

def generate_u3r_graph(num_vertices: int):
    """生成一个指定顶点数的3-正则图 (u3r)。"""
    if num_vertices % 2 != 0:
        raise ValueError("3-正则图的顶点数必须是偶数。")
    return nx.random_regular_graph(3, num_vertices)

def convert_graph_to_biqmac_string(graph: nx.Graph) -> str:
    """将 networkx 图对象转换为 BiqMac 格式的字符串。"""
    num_vertices = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    
    lines = [f"{num_vertices} {num_edges}"]
    for u, v in graph.edges():
        lines.append(f"{u + 1} {v + 1} 1")
    res = "\n".join(lines)
    with open(DATA_FILE, 'w') as f:
        f.write(res)
    return None
    
def save_dataset_entry(graph, exact_cut, partition, output_npy_file):
    """将图和解打包成字典并保存为.npy文件。"""
    if exact_cut is None or partition is None:
        print("    数据不完整，跳过保存。")
        return None

    dataset_entry = {
        "graph_adj_list": nx.to_dict_of_lists(graph),
        "num_vertices": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "exact_max_cut": exact_cut,
        "partition": np.array(partition)
    }
    np.save(output_npy_file, dataset_entry)
    print(f"    已将数据集条目保存到: '{output_npy_file}'")


# --- 主执行逻辑 ---
if __name__ == "__main__":
    # 定义数据集参数
    node_sizes = [40, 60, 80, 100, 120, 140]
    # node_sizes = [40]

    num_samples_per_size = 40
    main_dataset_dir = "u3r_dataset"

    # 创建主数据集目录
    os.makedirs(main_dataset_dir, exist_ok=True)

    # 主循环
    for n in node_sizes:
        size_dir = os.path.join(main_dataset_dir, f"n{n}")
        os.makedirs(size_dir, exist_ok=True)
        print(f"\n--- 开始处理顶点数为 {n} 的图 ---")
        
        for i in range(num_samples_per_size):
            print(f"\n  正在处理第 {i+1}/{num_samples_per_size} 个图...")
            
            # 定义文件名
            output_npy_path = os.path.join(size_dir, f"graph_{i}.npy")
            if os.path.exists(output_npy_path):
                print(f"    文件已存在，跳过: {output_npy_path}")
                continue

            # 生成图
            G = generate_u3r_graph(n)
            
            # 转换为BiqMac合法字符串并写入文件test.dat
            convert_graph_to_biqmac_string(G)
            
            # 3. 从test.dat提取输入，并提交到NEOS，等待获取结果
            job_num, job_pass = submit_to_neos()
            
            if job_num:
                results_str = poll_and_get_results(job_num, job_pass)
                if results_str:
                    # 4. 解析结果
                    cut_val, sol_vec = parse_neos_output_string(results_str)
                    
                    # 5. 保存数据
                    save_dataset_entry(G, cut_val, sol_vec, output_npy_path)
                else:
                    print("    未能从NEOS获取结果。")
            else:
                print("    任务提交失败，跳过此图。")
            
            #在两次API提交之间稍作停顿
            time.sleep(5) 
            
    print("\n--- 所有任务完成 ---")