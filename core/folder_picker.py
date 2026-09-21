"""Windows 原生文件夹选择对话框。

为什么单独一个模块：

* Gradio 的 ``gr.File`` / ``gr.Files`` 只能选择**文件**，无法选择"文件夹"；
* 第六轮任务书要求「输入文件夹 / 输出文件夹」都用
  **点击按钮 → 弹出 Windows 文件夹选择窗口**，并且**不允许用户手动输入路径**；
* 本工具是本地运行（服务端就在用户这台电脑上），所以在服务端弹出系统对话框
  就等价于在用户桌面上弹出对话框。

实现方式（**四级兜底，全部是 Windows 自带能力，不引入任何第三方依赖**）：

1. **现代 Windows 原生文件夹对话框**：直接通过 ``ctypes`` 调用 Shell 的
   ``IFileDialog``（``CLSID_FileOpenDialog`` + ``FOS_PICKFOLDERS``），
   也就是资源管理器里"选择文件夹"时看到的那个现代对话框
   （左侧导航栏、地址栏、搜索框、新建文件夹按钮，可调整大小）。
2. 老式但同样是**系统原生**的 ``SHBrowseForFolderW``
   （``BIF_NEWDIALOGSTYLE``：可调整大小、可新建文件夹的 Shell 目录树）。
3. PowerShell 的 ``Shell.Application.BrowseForFolder``（仍是系统 Shell 对话框）。
4. 最后才退回 tkinter（仅在前三种都不可用时；例如非 Windows 或无 GUI 环境）。

注意：上面没有一个是"自己画的文件浏览器"，都是系统对话框；只是风格新旧不同。

约定：函数**从不抛异常**。返回 ``None`` 表示"用户取消 / 没有可用对话框"，
调用方据此保持原有选择不变。
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import threading
import time
from typing import Optional, Tuple

from config import get_logger

logger = get_logger("ai_restore.folder")

#: 支持选择的图片扩展名（大小写不敏感）
IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.tif", "*.tiff")

#: 同一时刻只允许一个文件夹对话框（Gradio 回调是并发的）
_DIALOG_LOCK = threading.Lock()

#: 最近一次实际使用的后端（用于界面提示 / 排查："native" / "shbrowse" / "powershell" / "tkinter"）
_LAST_BACKEND: str = ""


def last_backend() -> str:
    """返回最近一次文件夹对话框实际使用的后端名字（自检/排查用）。"""
    return _LAST_BACKEND


_DPI_READY = False


def ensure_dpi_awareness() -> None:
    """让本进程声明 DPI 感知（幂等）。

    为什么必须做：如果进程是"DPI 不感知"的，Windows 会把系统对话框**整体位图拉伸**
    到当前缩放比例——在高分屏上就会变成"又大又糊、像假界面"。声明感知后，
    原生对话框会按真实 DPI 绘制，尺寸恢复正常并且清晰。
    """
    global _DPI_READY
    if _DPI_READY or not is_windows():
        return
    user32 = ctypes.windll.user32
    try:                       # Windows 10 1703+：Per-Monitor V2（最佳）
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 == -4
        if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            _DPI_READY = True
            return
    except Exception:
        pass
    try:                       # Windows 8.1+
        if ctypes.windll.shcore.SetProcessDpiAwareness(2) == 0:   # PROCESS_PER_MONITOR_DPI_AWARE
            _DPI_READY = True
            return
    except Exception:
        pass
    try:                       # 老系统兜底
        user32.SetProcessDPIAware()
    except Exception:
        pass
    _DPI_READY = True


def is_windows() -> bool:
    return os.name == "nt"


def normalize_dir(path: Optional[str]) -> str:
    """把用户/对话框返回的路径整理成绝对路径（不做存在性检查）。"""
    if not path:
        return ""
    raw = str(path).strip().strip('"').strip("'")
    if not raw:
        return ""
    return os.path.normpath(os.path.abspath(os.path.expanduser(raw)))


def tk_available() -> bool:
    """tkinter 是否可用（只做导入检查，不弹窗）。"""
    try:
        import tkinter  # noqa: F401
        from tkinter import filedialog  # noqa: F401
    except Exception:
        return False
    return True


def powershell_available() -> bool:
    if not is_windows():
        return False
    for exe in ("powershell.exe", "pwsh.exe"):
        try:
            out = subprocess.run(
                [exe, "-NoProfile", "-Command", "Write-Output ok"],
                capture_output=True, text=True, timeout=20,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if out.returncode == 0 and "ok" in (out.stdout or ""):
                return True
        except Exception:
            continue
    return False


def describe_capability() -> str:
    """一句话说明当前环境能用哪种方式选文件夹。"""
    if native_dialog_available():
        return "系统文件夹选择：可用（Windows 原生资源管理器风格对话框）"
    if is_windows():
        return "系统文件夹选择：可用（Windows 系统目录对话框 / PowerShell Shell）"
    if tk_available():
        return "系统文件夹选择：可用（tkinter 对话框）"
    return "系统文件夹选择：当前环境不可用"


# --------------------------------------------------------------------------
# 方案一：tkinter
# --------------------------------------------------------------------------
def _pick_tk(title: str, initial: Optional[str], must_exist: bool) -> Tuple[Optional[str], Optional[str]]:
    """返回 ``(路径, 错误)``；``路径`` 为 "" 表示用户取消。"""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - 环境相关
        return None, f"tkinter 不可用：{exc.__class__.__name__}"

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        root.update()
        kwargs = {"title": title, "mustexist": bool(must_exist)}
        start = normalize_dir(initial)
        if start and os.path.isdir(start):
            kwargs["initialdir"] = start
        path = filedialog.askdirectory(**kwargs)
        return (path or ""), None
    except Exception as exc:  # pragma: no cover - 环境相关
        return None, f"{exc.__class__.__name__}: {exc}"
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


# --------------------------------------------------------------------------
# 方案一：现代 Windows 原生文件夹对话框（IFileDialog + FOS_PICKFOLDERS）
# --------------------------------------------------------------------------
_CLSID_FileOpenDialog = "{DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7}"
_IID_IFileOpenDialog = "{D57C7288-D4AD-4768-BE02-9D969532D960}"
_IID_IShellItem = "{43826D1E-E718-42EE-BC55-A1E261C37BFE}"

_FOS_PICKFOLDERS = 0x00000020
_FOS_FORCEFILESYSTEM = 0x00000040
_FOS_PATHMUSTEXIST = 0x00000800
_SIGDN_FILESYSPATH = 0x80058000
_CLSCTX_INPROC_SERVER = 0x1
_COINIT_APARTMENTTHREADED = 0x2


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_ulong), ("Data2", ctypes.c_ushort),
                ("Data3", ctypes.c_ushort), ("Data4", ctypes.c_ubyte * 8)]

    @classmethod
    def parse(cls, text: str) -> "_GUID":
        s = text.strip().strip("{}").replace("-", "")
        return cls(int(s[0:8], 16), int(s[8:12], 16), int(s[12:16], 16),
                   (ctypes.c_ubyte * 8)(*[int(s[i:i + 2], 16) for i in range(16, 32, 2)]))


def _com_method(ptr, index: int, *argtypes):
    """取出 COM 接口第 index 个虚表函数（ctypes 通用做法）。"""
    vtbl = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    proto = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)
    return proto(vtbl[index])


def _com_release(ptr) -> None:
    if not ptr:
        return
    try:
        _com_method(ptr, 2)(ptr)
    except Exception:
        pass


def _windows_native_folder(title: str, initial: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """现代 Windows 原生文件夹选择对话框（IFileDialog + FOS_PICKFOLDERS）。

    全部走 ``ctypes`` 调用系统 Shell，等价于资源管理器里的"选择文件夹"对话框。
    """
    if not is_windows():
        return None, "当前系统不是 Windows"
    ensure_dpi_awareness()
    ole32 = ctypes.windll.ole32
    shell32 = ctypes.windll.shell32
    ole32.CoInitializeEx.restype = ctypes.c_long
    hr = ole32.CoInitializeEx(None, _COINIT_APARTMENTTHREADED)
    com_ready = hr in (0, 1)                     # S_OK / S_FALSE
    p_dialog = ctypes.c_void_p()
    p_item = ctypes.c_void_p()
    p_initial = ctypes.c_void_p()
    try:
        clsid = _GUID.parse(_CLSID_FileOpenDialog)
        iid = _GUID.parse(_IID_IFileOpenDialog)
        hr = ole32.CoCreateInstance(ctypes.byref(clsid), None, _CLSCTX_INPROC_SERVER,
                                    ctypes.byref(iid), ctypes.byref(p_dialog))
        if hr != 0 or not p_dialog:
            return None, f"CoCreateInstance 失败（0x{hr & 0xFFFFFFFF:08X}）"

        # GetOptions(10) → SetOptions(9)
        opts = ctypes.c_uint(0)
        _com_method(p_dialog, 10, ctypes.POINTER(ctypes.c_uint))(p_dialog, ctypes.byref(opts))
        new_opts = (opts.value | _FOS_PICKFOLDERS | _FOS_FORCEFILESYSTEM | _FOS_PATHMUSTEXIST)
        _com_method(p_dialog, 9, ctypes.c_uint)(p_dialog, new_opts)
        _com_method(p_dialog, 17, ctypes.c_wchar_p)(p_dialog, title or "选择文件夹")

        start = normalize_dir(initial)
        if start and os.path.isdir(start):
            try:
                iid_item = _GUID.parse(_IID_IShellItem)
                hr2 = shell32.SHCreateItemFromParsingName(
                    ctypes.c_wchar_p(start), None, ctypes.byref(iid_item),
                    ctypes.byref(p_initial))
                if hr2 == 0 and p_initial:
                    _com_method(p_dialog, 12, ctypes.c_void_p)(p_dialog, p_initial)
            except Exception as exc:                     # 初值失败不影响选择
                logger.debug("设置初始目录失败：%s", exc)

        hwnd = None
        try:
            hwnd = ctypes.windll.user32.GetForegroundWindow()
        except Exception:
            hwnd = None
        # 看护线程：系统对话框有时会以"最大化"出现（Windows 会记住上次尺寸），
        # 我们把它还原成正常小窗口——纯原生窗口操作，不是自己画的界面。
        stop_watch = threading.Event()
        watcher = threading.Thread(target=_watch_and_fit, args=(title or "选择文件夹", stop_watch),
                                   daemon=True)
        watcher.start()
        try:
            hr = _com_method(p_dialog, 3, ctypes.c_void_p)(p_dialog, hwnd)     # Show
        finally:
            stop_watch.set()
        if hr != 0:
            return "", None                            # 用户取消（HRESULT_FROM_WIN32(ERROR_CANCELLED)）

        hr = _com_method(p_dialog, 20, ctypes.POINTER(ctypes.c_void_p))(
            p_dialog, ctypes.byref(p_item))            # GetResult
        if hr != 0 or not p_item:
            return "", None
        buf = ctypes.c_wchar_p()
        hr = _com_method(p_item, 5, ctypes.c_uint, ctypes.POINTER(ctypes.c_wchar_p))(
            p_item, _SIGDN_FILESYSPATH, ctypes.byref(buf))
        if hr != 0 or not buf.value:
            return "", None
        path = str(buf.value)
        try:
            ole32.CoTaskMemFree(buf)
        except Exception:
            pass
        return path, None
    except Exception as exc:  # pragma: no cover - 环境相关
        return None, f"{exc.__class__.__name__}: {exc}"
    finally:
        _com_release(p_item)
        _com_release(p_initial)
        _com_release(p_dialog)
        if com_ready:
            try:
                ole32.CoUninitialize()
            except Exception:
                pass


# --------------------------------------------------------------------------
# 让系统对话框以"正常小窗口"出现（不自己画界面，只做原生窗口尺寸/位置调整）
# --------------------------------------------------------------------------
#: 目标窗口占工作区的比例（宽度、高度）。系统默认有时会记住"最大化"，看起来像全屏。
DIALOG_WORKAREA_RATIO = (0.62, 0.66)


def _visible_windows():
    """枚举当前可见的顶层窗口：``(hwnd, 标题, 类名)``。"""
    user32 = ctypes.windll.user32
    out = []

    def cb(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        cls = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, cls, 256)
        out.append((int(hwnd), buf.value, cls.value))
        return True

    try:
        EnumProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        user32.EnumWindows(EnumProc(cb), None)
    except Exception:
        pass
    return out


def _fit_dialog_native(hwnd: int, title: str) -> Optional[Tuple[int, int, int, int]]:
    """把系统对话框还原成"正常小窗口"并居中（仅原生窗口操作）。

    ``Show()`` 期间由看护线程调用：如果窗口处于最大化（Windows 会记住上一次的尺寸，
    有时就是最大化），先 ``SW_RESTORE``，再按工作区的固定比例摆放并居中。
    返回 ``(x, y, w, h)`` 便于日志/自检。
    """
    user32 = ctypes.windll.user32
    SW_RESTORE = 9
    SPI_GETWORKAREA = 0x0030
    SWP_NOZORDER, SWP_NOACTIVATE, SWP_SHOWWINDOW = 0x0004, 0x0010, 0x0040
    try:
        if user32.IsZoomed(hwnd) or user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, SW_RESTORE)

        class RECT(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                        ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

        wa = RECT()
        if not user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(wa), 0):
            wa.left, wa.top = 0, 0
            wa.right, wa.bottom = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        avail_w = max(640, wa.right - wa.left)
        avail_h = max(480, wa.bottom - wa.top)
        w = int(max(760, min(avail_w - 40, avail_w * DIALOG_WORKAREA_RATIO[0])))
        h = int(max(520, min(avail_h - 40, avail_h * DIALOG_WORKAREA_RATIO[1])))
        x = wa.left + (avail_w - w) // 2
        y = wa.top + (avail_h - h) // 2
        user32.SetWindowPos(hwnd, 0, x, y, w, h,
                            SWP_NOZORDER | SWP_NOACTIVATE | SWP_SHOWWINDOW)
        user32.SetForegroundWindow(hwnd)
        logger.info("文件夹对话框已调整为小窗口：%d×%d（工作区 %d×%d）", w, h, avail_w, avail_h)
        return (x, y, w, h)
    except Exception as exc:
        logger.warning("调整对话框尺寸失败：%s", exc)
        return None


def _watch_and_fit(title: str, stop: threading.Event, timeout_s: float = 20.0) -> None:
    """看护线程：等对话框出现 → 调整成小窗口 → 结束（找窗口失败也不影响主流程）。"""
    deadline = time.time() + float(timeout_s)
    while not stop.is_set() and time.time() < deadline:
        for hwnd, win_title, cls in _visible_windows():
            if cls != "#32770":
                continue
            if win_title == title or (title and title in win_title):
                _fit_dialog_native(hwnd, win_title)
                return
        time.sleep(0.12)


def native_dialog_available() -> bool:
    """探测现代原生文件夹对话框能否创建（**不弹窗**，用于自检与测试）。"""
    if not is_windows():
        return False
    ole32 = ctypes.windll.ole32
    hr = ole32.CoInitializeEx(None, _COINIT_APARTMENTTHREADED)
    ready = hr in (0, 1)
    p = ctypes.c_void_p()
    try:
        clsid = _GUID.parse(_CLSID_FileOpenDialog)
        iid = _GUID.parse(_IID_IFileOpenDialog)
        hr = ole32.CoCreateInstance(ctypes.byref(clsid), None, _CLSCTX_INPROC_SERVER,
                                    ctypes.byref(iid), ctypes.byref(p))
        if hr != 0 or not p:
            return False
        opts = ctypes.c_uint(0)
        _com_method(p, 10, ctypes.POINTER(ctypes.c_uint))(p, ctypes.byref(opts))
        _com_method(p, 9, ctypes.c_uint)(p, opts.value | _FOS_PICKFOLDERS)
        _com_method(p, 17, ctypes.c_wchar_p)(p, "自检")
        return True
    except Exception:
        return False
    finally:
        _com_release(p)
        if ready:
            try:
                ole32.CoUninitialize()
            except Exception:
                pass


# --------------------------------------------------------------------------
# 方案二：老式原生 Shell 目录树（SHBrowseForFolder，BIF_NEWDIALOGSTYLE）
# --------------------------------------------------------------------------
class _BROWSEINFOW(ctypes.Structure):
    _fields_ = [("hwndOwner", ctypes.c_void_p), ("pidlRoot", ctypes.c_void_p),
                ("pszDisplayName", ctypes.c_wchar_p), ("lpszTitle", ctypes.c_wchar_p),
                ("ulFlags", ctypes.c_uint), ("lpfn", ctypes.c_void_p),
                ("lParam", ctypes.c_void_p), ("iImage", ctypes.c_int)]


_BIF_RETURNONLYFSDIRS = 0x0001
_BIF_NEWDIALOGSTYLE = 0x0040
_BIF_EDITBOX = 0x0010


def _shbrowse_folder(title: str, initial: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not is_windows():
        return None, "当前系统不是 Windows"
    ensure_dpi_awareness()
    ole32 = ctypes.windll.ole32
    shell32 = ctypes.windll.shell32
    hr = ole32.CoInitializeEx(None, _COINIT_APARTMENTTHREADED)
    ready = hr in (0, 1)
    pidl_root = ctypes.c_void_p()
    try:
        start = normalize_dir(initial)
        if start and os.path.isdir(start):
            try:
                shell32.SHParseDisplayName(ctypes.c_wchar_p(start), None,
                                           ctypes.byref(pidl_root), 0, None)
            except Exception:
                pidl_root = ctypes.c_void_p()
        name_buf = ctypes.create_unicode_buffer(512)
        bi = _BROWSEINFOW()
        bi.hwndOwner = None
        bi.pidlRoot = pidl_root
        bi.pszDisplayName = ctypes.cast(name_buf, ctypes.c_wchar_p)
        bi.lpszTitle = title or "选择文件夹"
        bi.ulFlags = _BIF_RETURNONLYFSDIRS | _BIF_NEWDIALOGSTYLE | _BIF_EDITBOX
        shell32.SHBrowseForFolderW.restype = ctypes.c_void_p
        pidl = shell32.SHBrowseForFolderW(ctypes.byref(bi))
        if not pidl:
            return "", None
        buf = ctypes.create_unicode_buffer(32768)
        ok = shell32.SHGetPathFromIDListW(ctypes.c_void_p(pidl), buf)
        ole32.CoTaskMemFree(ctypes.c_void_p(pidl))
        return (str(buf.value), None) if ok else ("", None)
    except Exception as exc:  # pragma: no cover - 环境相关
        return None, f"{exc.__class__.__name__}: {exc}"
    finally:
        if pidl_root:
            try:
                ole32.CoTaskMemFree(pidl_root)
            except Exception:
                pass
        if ready:
            try:
                ole32.CoUninitialize()
            except Exception:
                pass


# --------------------------------------------------------------------------
# 方案三：PowerShell Shell.Application（仍是系统 Shell 对话框）
# --------------------------------------------------------------------------
_PS_SCRIPT = """
$shell = New-Object -ComObject Shell.Application
$folder = $shell.BrowseForFolder(0, $env:AI_RESTORE_FOLDER_TITLE, 0, $env:AI_RESTORE_FOLDER_INITIAL)
if ($folder -ne $null) { [Console]::Out.Write($folder.Self.Path) }
"""
def _pick_shell_powershell(title: str, initial: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """PowerShell + Shell.Application.BrowseForFolder（系统 Shell 目录对话框）。"""
    if not is_windows():
        return None, "当前系统不是 Windows"
    env = dict(os.environ)
    env["AI_RESTORE_FOLDER_TITLE"] = title
    env["AI_RESTORE_FOLDER_INITIAL"] = normalize_dir(initial) or "0"
    last_err = "未找到可用的 PowerShell"
    for exe in ("powershell.exe", "pwsh.exe"):
        try:
            out = subprocess.run(
                [exe, "-NoProfile", "-STA", "-Command", _PS_SCRIPT],
                capture_output=True, text=True, timeout=600, env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception as exc:  # pragma: no cover - 环境相关
            last_err = f"{exc.__class__.__name__}: {exc}"
            continue
        if out.returncode == 0:
            return (out.stdout or "").strip(), None
        last_err = (out.stderr or "").strip()[:200] or f"退出码 {out.returncode}"
    return None, last_err


def _pick_powershell(title: str, initial: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not is_windows():
        return None, "当前系统不是 Windows"
    env = dict(os.environ)
    env["AI_RESTORE_FOLDER_TITLE"] = title
    env["AI_RESTORE_FOLDER_INITIAL"] = normalize_dir(initial)
    last_err = "未找到可用的 PowerShell"
    for exe in ("powershell.exe", "pwsh.exe"):
        try:
            out = subprocess.run(
                [exe, "-NoProfile", "-STA", "-Command", _PS_SCRIPT],
                capture_output=True, text=True, timeout=600, env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception as exc:  # pragma: no cover - 环境相关
            last_err = f"{exc.__class__.__name__}: {exc}"
            continue
        if out.returncode == 0:
            return (out.stdout or "").strip(), None
        last_err = (out.stderr or "").strip()[:200] or f"退出码 {out.returncode}"
    return None, last_err


# --------------------------------------------------------------------------
# 对外接口
# --------------------------------------------------------------------------
def pick_folder(title: str = "选择文件夹", initial: Optional[str] = None,
                must_exist: bool = True) -> Optional[str]:
    """弹出 Windows 原生文件夹选择窗口。

    * 用户选了目录 → 返回规范化后的绝对路径；
    * 用户点了取消 / 当前环境无法弹窗 → 返回 ``None``。
    """
    with _DIALOG_LOCK:
        global _LAST_BACKEND
        # ① 现代 Windows 原生对话框（IFileDialog + FOS_PICKFOLDERS）
        path, err = _windows_native_folder(title, initial)
        if path is None:
            logger.warning("现代原生文件夹对话框不可用（%s），改用 SHBrowseForFolder", err)
            # ② 老式但同样是系统原生的 Shell 目录树
            path, err2 = _shbrowse_folder(title, initial)
            if path is None:
                logger.warning("SHBrowseForFolder 不可用（%s），改用 PowerShell Shell", err2)
                # ③ PowerShell Shell.Application
                path, err3 = _pick_shell_powershell(title, initial)
                if path is None:
                    logger.warning("PowerShell Shell 也不可用（%s），最后退回 tkinter", err3)
                    # ④ 兜底：tkinter（仅前面都不可用时）
                    path, _err4 = _pick_tk(title, initial, must_exist)
                    _LAST_BACKEND = "tkinter"
                    if path is None:
                        return None
                else:
                    _LAST_BACKEND = "powershell"
            else:
                _LAST_BACKEND = "shbrowse"
        else:
            _LAST_BACKEND = "native"
        path = normalize_dir(path)
        return path or None


def pick_folder_safe(title: str = "选择文件夹", initial: Optional[str] = None,
                     must_exist: bool = True) -> Tuple[Optional[str], Optional[str]]:
    """与 :func:`pick_folder` 相同，但额外返回错误说明（供界面提示）。"""
    try:
        return pick_folder(title=title, initial=initial, must_exist=must_exist), None
    except Exception as exc:  # pragma: no cover - 兜底
        logger.warning("文件夹选择失败：%s", exc)
        return None, f"{exc.__class__.__name__}: {exc}"


# --------------------------------------------------------------------------
# 选择图片文件（能直接看到文件夹里的图片文件名）
# --------------------------------------------------------------------------
def pick_files(title: str = "选择图片（可多选）", initial: Optional[str] = None
               ) -> Tuple[List[str], Optional[str]]:
    """弹出**文件**选择对话框，可以直观看到文件夹里的图片。

    为什么需要它：``askdirectory`` 是"目录树"对话框，**只显示文件夹、不显示文件**，
    用户无法确认某个文件夹里到底有没有图片。``askopenfilenames`` 则直接列出文件名，
    并且支持多选。

    返回 ``(文件路径列表, 错误)``；用户取消时返回空列表（不是错误）。
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - 环境相关
        return [], f"tkinter 不可用：{exc.__class__.__name__}"

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        root.update()
        kwargs: dict = {"title": title, "filetypes": [("图片", " ".join(IMAGE_PATTERNS)),
                                                      ("所有文件", "*.*")]}
        start = normalize_dir(initial)
        if start and os.path.isdir(start):
            kwargs["initialdir"] = start
        paths = filedialog.askopenfilenames(**kwargs)
        return [str(p) for p in (paths or [])], None
    except Exception as exc:  # pragma: no cover - 环境相关
        return [], f"{exc.__class__.__name__}: {exc}"
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


def summarize_images(paths) -> str:
    """把一组图片路径整理成"数量 + 前几个文件名"的中文说明。"""
    names = [os.path.basename(str(p)) for p in (paths or [])]
    if not names:
        return "该文件夹中未找到支持的图片文件"
    head = "、".join(names[:5]) + ("…" if len(names) > 5 else "")
    return f"已找到 {len(names)} 张图片：{head}"


def ensure_writable_dir(path: str) -> Tuple[bool, str]:
    """检查（必要时创建）目录，并验证可写。返回 ``(是否可用, 说明)``。"""
    target = normalize_dir(path)
    if not target:
        return False, "路径为空"
    try:
        os.makedirs(target, exist_ok=True)
    except Exception as exc:
        return False, f"无法创建目录：{exc.__class__.__name__}: {exc}"
    probe = os.path.join(target, ".ai_restore_write_test")
    try:
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("ok")
        os.remove(probe)
    except Exception as exc:
        return False, f"目录不可写（可能没有权限）：{exc.__class__.__name__}: {exc}"
    return True, "目录可用"


if __name__ == "__main__":  # 自检：python -m core.folder_picker
    print(describe_capability())
    print("tk:", tk_available(), "powershell:", powershell_available())
