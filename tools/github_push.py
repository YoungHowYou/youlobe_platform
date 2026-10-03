# -*- coding: utf-8 -*-
"""
推送到 GitHub（自动绕过不可达的 GitHub IP）
==========================================
本机（局域网 192.168.3.x）存在一个网络问题：DNS 把 `github.com` 解析到
`20.205.243.166`，该地址的 443 连不上；而 GitHub 的其它 IP
（140.82.112.3 / 113.3 / 114.3）以及 `github.com:22` 都是通的，
PyPI 等其它站点也正常——也就是**只有 github.com 的这个解析结果被挡**。

本工具的做法：
  1. 先直连 github.com:443 试探，通就直接 `git push`；
  2. 不通则探测备用 IP，在本地起一个极简 HTTP CONNECT 代理把流量转发过去，
     再用 `git -c http.proxy=...` 推送（一次性参数）。

**不改 hosts、不需要管理员、不写全局 git 配置**，推完自动收摊。

用法（项目根目录）：
    venv\\Scripts\\python.exe tools\\github_push.py                # 推到 origin 当前分支
    venv\\Scripts\\python.exe tools\\github_push.py --check        # 只测连通性，不推送
    venv\\Scripts\\python.exe tools\\github_push.py --branch main --remote origin

若哪天 DNS 恢复正常、直连也能通，脚本会自动走直连（备用逻辑失效也无妨）。
"""

import argparse
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

HOST = "github.com"
PORT = 443
FALLBACK_IPS = ["140.82.113.3", "140.82.112.3", "140.82.114.3"]
PROXY_PORT = 8899


# --------------------------------------------------------------------------
# 连通性探测
# --------------------------------------------------------------------------
def tcp_ok(host: str, port: int = PORT, timeout: float = 5.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def pick_ip() -> str | None:
    """返回第一个可达的备用 IP；都不通返回 None。"""
    for ip in FALLBACK_IPS:
        if tcp_ok(ip):
            return ip
    return None


# --------------------------------------------------------------------------
# 极简 CONNECT 代理（仅本地回环，用完即停）
# --------------------------------------------------------------------------
def _pipe(a, b):
    import select

    try:
        while True:
            readable, _, _ = select.select([a, b], [], [], 300)
            for sock in readable:
                data = sock.recv(65536)
                if not data:
                    return
                (b if sock is a else a).sendall(data)
    except OSError:
        pass
    finally:
        for sock in (a, b):
            try:
                sock.close()
            except OSError:
                pass


def _serve(server_sock: socket.socket, dest_ip: str):
    while True:
        try:
            conn, _ = server_sock.accept()
        except OSError:
            return
        threading.Thread(target=_handle, args=(conn, dest_ip), daemon=True).start()


def _handle(conn: socket.socket, dest_ip: str):
    try:
        conn.settimeout(20)
        req = b""
        while b"\r\n\r\n" not in req:
            chunk = conn.recv(4096)
            if not chunk:
                conn.close()
                return
            req += chunk
        line = req.split(b"\r\n", 1)[0].decode("latin-1")
        method, target, _ = line.split(" ", 2)
        if method.upper() != "CONNECT":
            conn.sendall(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
            conn.close()
            return
        host, _, port = target.partition(":")
        dest = dest_ip if host.lower() == HOST else host
        upstream = socket.create_connection((dest, int(port or PORT)), timeout=20)
        conn.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
        conn.settimeout(None)
        upstream.settimeout(None)
        threading.Thread(target=_pipe, args=(conn, upstream), daemon=True).start()
    except OSError:
        try:
            conn.close()
        except OSError:
            pass


class LocalProxy:
    def __init__(self, dest_ip: str, port: int = PROXY_PORT):
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", port))
        self.sock.listen(50)
        threading.Thread(target=_serve, args=(self.sock, dest_ip), daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.sock.getsockname()[1]}"

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


# --------------------------------------------------------------------------
# git 操作
# --------------------------------------------------------------------------
def git(*args, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(ROOT), text=True,
                          encoding="utf-8", errors="ignore", **kw)


def current_branch() -> str:
    res = git("rev-parse", "--abbrev-ref", "HEAD", capture_output=True)
    return (res.stdout or "").strip() or "main"


def main() -> int:
    parser = argparse.ArgumentParser(description="推送到 GitHub（自动绕过不可达 IP）")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default=None, help="默认取当前分支")
    parser.add_argument("--check", action="store_true", help="只测连通性")
    args = parser.parse_args()

    branch = args.branch or current_branch()
    print(f"分支：{branch}    远端：{args.remote}")

    direct = tcp_ok(HOST)
    print(f"直连 {HOST}:{PORT} → {'通' if direct else '不通'}")

    proxy = None
    push_args: list = []
    if direct:
        print("走直连推送")
    else:
        ip = pick_ip()
        if not ip:
            print("备用 GitHub IP 也不可达。请检查网络/代理，或改用 SSH 远端。")
            return 2
        print(f"备用 IP {ip} 可达 → 起本地转发 {PROXY_PORT}")
        proxy = LocalProxy(ip)
        time.sleep(0.3)
        push_args = ["-c", f"http.proxy={proxy.url}"]
        print(f"经 {proxy.url} 推送")

    try:
        if args.check:
            print("\n（--check，未推送）")
            return 0

        res = git(*push_args, "push", args.remote, branch)
        if res.returncode != 0:
            print("\n推送失败。若提示认证/权限，请在能联网的机器上执行，"
                  "或先配置好凭据（git credential manager）。")
            return res.returncode

        print("\n已推送。")
        return 0
    finally:
        if proxy is not None:
            proxy.close()
            print("本地转发已关闭（未修改 hosts、未写全局 git 配置）")


if __name__ == "__main__":
    raise SystemExit(main())
