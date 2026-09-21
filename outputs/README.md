# outputs

修复结果、对比图与 Mask 图统一保存在此目录：

* `restored_YYYYmmdd_HHMMSS.png` —— AI 修复结果
* `compare_YYYYmmdd_HHMMSS.png` —— 原图 / 修复结果 左右对比图
* `mask_YYYYmmdd_HHMMSS.png` —— 本次实际用于修复的 Mask
* `test/` —— `tests/test_all.py` 自动测试生成的结果图

此目录内容可以随时删除，不影响程序运行。
