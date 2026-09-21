# -*- coding: utf-8 -*-
"""
Antigravity 会话一键导出脚本
将当前车辆识别会话 (ID: b61e6f45-d002-412f-b4ed-a508c34a4c70)
打包为可移植的 zip 压缩包，方便迁移到笔记本。
"""

import os
import sys
import shutil
import sqlite3
import zipfile
import tempfile

CONV_ID = "b61e6f45-d002-412f-b4ed-a508c34a4c70"
USER_HOME = os.path.expanduser("~")
ANTIGRAVITY_DIR = os.path.join(USER_HOME, ".gemini", "antigravity")

IMPORT_SCRIPT_CONTENT = '''# -*- coding: utf-8 -*-
"""
Antigravity 会话一键导入脚本 (在笔记本上运行)
将导出的会话自动写入笔记本的 Antigravity 数据库与 brain 目录。
"""

import os
import sys
import shutil
import sqlite3
import zipfile

CONV_ID = "b61e6f45-d002-412f-b4ed-a508c34a4c70"
USER_HOME = os.path.expanduser("~")
TARGET_AG_DIR = os.path.join(USER_HOME, ".gemini", "antigravity")

def import_session():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_db = os.path.join(current_dir, f"{CONV_ID}.db")
    src_brain = os.path.join(current_dir, "brain")

    if not os.path.exists(src_db):
        print(f"[!] 错误: 未找到数据库文件: {src_db}")
        return False

    os.makedirs(os.path.join(TARGET_AG_DIR, "conversations"), exist_ok=True)
    os.makedirs(os.path.join(TARGET_AG_DIR, "brain"), exist_ok=True)

    # 1. 拷贝会话数据库
    dst_db = os.path.join(TARGET_AG_DIR, "conversations", f"{CONV_ID}.db")
    print(f"[*] 正在导入会话数据库至: {dst_db}")
    shutil.copy2(src_db, dst_db)

    # 2. 拷贝 brain 目录
    dst_brain = os.path.join(TARGET_AG_DIR, "brain", CONV_ID)
    if os.path.exists(src_brain):
        print(f"[*] 正在导入脑日志与工件至: {dst_brain}")
        if os.path.exists(dst_brain):
            shutil.rmtree(dst_brain)
        shutil.copytree(src_brain, dst_brain)

    # 3. 注册到笔记本的 conversation_summaries.db (左侧历史列表)
    src_summary_db = os.path.join(current_dir, "summary_export.db")
    target_summary_db = os.path.join(TARGET_AG_DIR, "conversation_summaries.db")

    if os.path.exists(src_summary_db) and os.path.exists(target_summary_db):
        try:
            print("[*] 正在向笔记本左侧历史列表注册会话摘要...")
            s_con = sqlite3.connect(src_summary_db)
            s_cur = s_con.cursor()
            row = s_cur.execute("SELECT * FROM conversation_summaries WHERE conversation_id = ?", (CONV_ID,)).fetchone()
            col_names = [d[0] for d in s_cur.description]
            s_con.close()

            if row:
                t_con = sqlite3.connect(target_summary_db)
                t_cur = t_con.cursor()
                placeholders = ", ".join(["?"] * len(col_names))
                cols_str = ", ".join(col_names)
                t_cur.execute(f"INSERT OR REPLACE INTO conversation_summaries ({cols_str}) VALUES ({placeholders})", row)
                t_con.commit()
                t_con.close()
                print("[*] 会话摘要注册成功！")
        except Exception as e:
            print(f"[!] 警告: 摘要注册出现小提示 (不影响会话打开): {e}")

    print("\\n" + "=" * 60)
    print("[*] 恭喜！会话迁移导入全部完成！")
    print("[*] 请在笔记本上退出并重新打开 Antigravity，即可在左侧历史记录中看到该会话并继续对话。")
    print("=" * 60 + "\\n")
    return True

if __name__ == "__main__":
    import_session()
'''


def export_session(output_zip_path=None):
    if not output_zip_path:
        desktop = os.path.join(USER_HOME, "Desktop")
        output_zip_path = os.path.join(desktop, f"antigravity_session_{CONV_ID[:8]}.zip")

    src_db = os.path.join(ANTIGRAVITY_DIR, "conversations", f"{CONV_ID}.db")
    src_brain = os.path.join(ANTIGRAVITY_DIR, "brain", CONV_ID)
    summary_db = os.path.join(ANTIGRAVITY_DIR, "conversation_summaries.db")

    if not os.path.exists(src_db):
        print(f"[!] 错误: 未在本地找到该会话数据库: {src_db}")
        return False

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. 干净导出 SQLite 会话数据库 (利用 online backup api 避免锁冲突)
        dst_db = os.path.join(tmpdir, f"{CONV_ID}.db")
        print(f"[*] 正在打包提取会话消息数据库...")
        s_conn = sqlite3.connect(src_db)
        d_conn = sqlite3.connect(dst_db)
        s_conn.backup(d_conn)
        s_conn.close()
        d_conn.close()

        # 2. 导出会话摘要信息
        if os.path.exists(summary_db):
            print(f"[*] 正在打包提取会话左侧列表摘要元数据...")
            sum_export_db = os.path.join(tmpdir, "summary_export.db")
            s_sum = sqlite3.connect(summary_db)
            d_sum = sqlite3.connect(sum_export_db)
            s_sum.backup(d_sum)
            s_sum.close()
            d_sum.close()

        # 3. 导出 brain 目录 (包含工件与推理轨迹)
        if os.path.exists(src_brain):
            print(f"[*] 正在打包提取会话思维链、代码工件与执行日志...")
            shutil.copytree(src_brain, os.path.join(tmpdir, "brain"))

        # 4. 写入笔记本一键导入脚本
        with open(os.path.join(tmpdir, "一键导入到笔记本.py"), "w", encoding="utf-8") as f:
            f.write(IMPORT_SCRIPT_CONTENT)

        # 写入双击运行的 bat
        with open(os.path.join(tmpdir, "一键导入到笔记本.bat"), "w", encoding="gbk") as f:
            f.write('@echo off\npython "一键导入到笔记本.py"\npause\n')

        # 5. 压缩为最终 ZIP 包
        print(f"[*] 正在压缩生成最终迁移文件: {output_zip_path}")
        with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    full_p = os.path.join(root, file)
                    rel_p = os.path.relpath(full_p, tmpdir)
                    zf.write(full_p, rel_p)

    file_size_mb = os.path.getsize(output_zip_path) / (1024 * 1024)
    print("\n" + "=" * 60)
    print(f"[*] 导出成功！会话安装包已生成在桌面上:")
    print(f"    文件路径: {os.path.abspath(output_zip_path)}")
    print(f"    文件大小: {file_size_mb:.2f} MB")
    print("=" * 60 + "\n")
    return True


if __name__ == "__main__":
    export_session()
