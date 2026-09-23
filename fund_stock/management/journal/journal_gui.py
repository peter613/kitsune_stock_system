"""
journal_gui.py
Kitsune Kernel — 交易紀錄快速輸入系統 (Tkinter GUI)
位置：fund_stock/management/journal/journal_gui.py
"""

import sys
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox

# 專案路徑設定
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))
import config

from fund_stock.management.journal.update_journal import (
    get_latest_cycle,
    load_stock_names,
    append_journal_entry,
    delete_last_entry,
    get_recent_entries,
    REASONS
)


class JournalApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("小狐狸量化 — 交易紀錄輸入系統 (Kitsune Journal)")
        self.geometry("880x700")
        self.minsize(820, 640)

        # 載入白名單股票名稱
        self.stock_names = load_stock_names()

        # 變數綁定
        self.cycle_var = tk.IntVar(value=get_latest_cycle())
        self.date_var = tk.StringVar(value=datetime.now().strftime("%Y/%m/%d"))
        self.stock_id_var = tk.StringVar()
        self.stock_name_var = tk.StringVar(value="請輸入代號")
        self.action_var = tk.StringVar(value="buy")
        self.share_var = tk.StringVar()
        self.price_var = tk.StringVar()
        self.reason_var = tk.StringVar(value="ENTRY")
        self.status_var = tk.StringVar(value="系統就緒")

        # 監聽股票代號變化以即時查名稱
        self.stock_id_var.trace_add("write", self._on_stock_id_changed)
        # 監聽買賣切換以提供預設原因
        self.action_var.trace_add("write", self._on_action_changed)

        self._setup_style()
        self._create_widgets()
        self.refresh_table()

    def _setup_style(self):
        self.style = ttk.Style(self)
        # 使用原生或乾淨主題
        available_themes = self.style.theme_names()
        if "clam" in available_themes:
            self.style.theme_use("clam")

        # 設定字型與顏色 (放大輸入欄位字型)
        font_family = "Microsoft JhengHei" if sys.platform == "win32" else "Helvetica"
        self.input_font = (font_family, 13)
        self.label_font = (font_family, 11, "bold")
        self.small_font = (font_family, 10)
        self.stock_name_font = (font_family, 13, "bold")
        self.title_font = (font_family, 15, "bold")
        self.button_font = (font_family, 11, "bold")
        self.tree_font = (font_family, 10)
        self.tree_head_font = (font_family, 10, "bold")

        # 套用至 Combobox 下拉清單字型
        self.option_add("*TCombobox*Listbox.font", self.input_font)

        self.style.configure(".", font=self.input_font)
        self.style.configure("FormLabel.TLabel", font=self.label_font, foreground="#2C3E50")
        self.style.configure("Hint.TLabel", font=self.small_font, foreground="#7F8C8D")
        self.style.configure("Title.TLabel", font=self.title_font, foreground="#1A2B4C")
        self.style.configure("StockName.TLabel", font=self.stock_name_font, foreground="#2C3E50")
        self.style.configure("TEntry", font=self.input_font)
        self.style.configure("TSpinbox", font=self.input_font)
        self.style.configure("TCombobox", font=self.input_font)
        self.style.configure("TRadiobutton", font=self.input_font)
        self.style.configure("TButton", font=self.button_font)
        self.style.configure("Accent.TButton", font=self.button_font, padding=6)
        self.style.configure("Danger.TButton", font=self.button_font, padding=4)
        self.style.configure("Treeview.Heading", font=self.tree_head_font)
        self.style.configure("Treeview", font=self.tree_font, rowheight=26)

    def _create_widgets(self):
        # 主容器
        main_frame = ttk.Frame(self, padding="15 15 15 10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 頂部標題
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(title_frame, text="台股交易紀錄輸入 (journal.csv)", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(title_frame, text="🦊 Kitsune Journal", style="Hint.TLabel").pack(side=tk.RIGHT)

        # 表單外框
        form_frame = ttk.LabelFrame(main_frame, text=" 交易資訊輸入 ", padding="15 12 15 12")
        form_frame.pack(fill=tk.X, pady=(0, 10))

        # Grid 配置: 4 列
        # 第 0 列: 週期 & 日期
        ttk.Label(form_frame, text="換股週期 (cycle):", style="FormLabel.TLabel").grid(row=0, column=0, sticky=tk.W, pady=6)
        cycle_box = ttk.Frame(form_frame)
        cycle_box.grid(row=0, column=1, sticky=tk.W, pady=6)
        self.cycle_entry = ttk.Spinbox(cycle_box, from_=0, to=999, textvariable=self.cycle_var, width=6, font=self.input_font)
        self.cycle_entry.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(cycle_box, text="+1 新週期", width=9, command=self._increment_cycle).pack(side=tk.LEFT, padx=2)
        ttk.Button(cycle_box, text="回到最新", width=8, command=self._reset_cycle).pack(side=tk.LEFT, padx=2)

        ttk.Label(form_frame, text="交易日期 (date):", style="FormLabel.TLabel").grid(row=0, column=2, sticky=tk.W, padx=(20, 0), pady=6)
        date_box = ttk.Frame(form_frame)
        date_box.grid(row=0, column=3, sticky=tk.W, pady=6)
        ttk.Entry(date_box, textvariable=self.date_var, width=12, font=self.input_font).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(date_box, text="今日", width=5, command=self._set_today).pack(side=tk.LEFT)

        # 第 1 列: 股票代號 & 名稱反查
        ttk.Label(form_frame, text="股票代號 (id):", style="FormLabel.TLabel").grid(row=1, column=0, sticky=tk.W, pady=6)
        id_box = ttk.Frame(form_frame)
        id_box.grid(row=1, column=1, sticky=tk.W, pady=6)
        self.id_entry = ttk.Entry(id_box, textvariable=self.stock_id_var, width=10, font=self.input_font)
        self.id_entry.pack(side=tk.LEFT, padx=(0, 8))
        self.lbl_stock_name = ttk.Label(id_box, textvariable=self.stock_name_var, style="StockName.TLabel")
        self.lbl_stock_name.pack(side=tk.LEFT)

        ttk.Label(form_frame, text="買賣動作 (action):", style="FormLabel.TLabel").grid(row=1, column=2, sticky=tk.W, padx=(20, 0), pady=6)
        act_box = ttk.Frame(form_frame)
        act_box.grid(row=1, column=3, sticky=tk.W, pady=6)
        ttk.Radiobutton(act_box, text="買進 (buy)", variable=self.action_var, value="buy").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(act_box, text="賣出 (sell)", variable=self.action_var, value="sell").pack(side=tk.LEFT)

        # 第 2 列: 股數 & 單價
        ttk.Label(form_frame, text="成交股數 (share):", style="FormLabel.TLabel").grid(row=2, column=0, sticky=tk.W, pady=6)
        share_box = ttk.Frame(form_frame)
        share_box.grid(row=2, column=1, sticky=tk.W, pady=6)
        ttk.Entry(share_box, textvariable=self.share_var, width=10, font=self.input_font).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(share_box, text="1張", width=4, command=lambda: self.share_var.set("1000")).pack(side=tk.LEFT, padx=1)
        ttk.Button(share_box, text="半張", width=4, command=lambda: self.share_var.set("500")).pack(side=tk.LEFT, padx=1)

        ttk.Label(form_frame, text="成交單價 (price):", style="FormLabel.TLabel").grid(row=2, column=2, sticky=tk.W, padx=(20, 0), pady=6)
        ttk.Entry(form_frame, textvariable=self.price_var, width=12, font=self.input_font).grid(row=2, column=3, sticky=tk.W, pady=6)

        # 第 3 列: 交易原因
        ttk.Label(form_frame, text="交易原因 (reason):", style="FormLabel.TLabel").grid(row=3, column=0, sticky=tk.W, pady=6)
        self.reason_combo = ttk.Combobox(form_frame, textvariable=self.reason_var, values=REASONS, state="readonly", width=16, font=self.input_font)
        self.reason_combo.grid(row=3, column=1, sticky=tk.W, pady=6)

        reason_hint_lbl = ttk.Label(form_frame, text="ENTRY=買進 | REBALANCE=換股 | SL_HALF=砍半 | SL_FULL=砍全 | TP_QUARTER=停利1/4", style="Hint.TLabel")
        reason_hint_lbl.grid(row=3, column=2, columnspan=2, sticky=tk.W, padx=(20, 0), pady=6)

        # 功能按鈕列
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        btn_submit = ttk.Button(btn_frame, text="✔ 確認寫入交易", style="Accent.TButton", command=self.submit_entry)
        btn_submit.pack(side=tk.LEFT, padx=(0, 8))

        btn_clear = ttk.Button(btn_frame, text="清空輸入", command=self.clear_inputs)
        btn_clear.pack(side=tk.LEFT, padx=(0, 8))

        btn_undo = ttk.Button(btn_frame, text="復原/刪除最後一筆", style="Danger.TButton", command=self.undo_last)
        btn_undo.pack(side=tk.RIGHT)

        # 表格區域 (近期紀錄)
        table_frame = ttk.LabelFrame(main_frame, text=" 最近交易紀錄 (journal.csv) ", padding="5")
        table_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("cycle", "date", "id", "name", "action", "share", "price", "reason")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)

        # 欄位標題與寬度
        col_configs = {
            "cycle": ("週期", 55, tk.CENTER),
            "date": ("日期", 90, tk.CENTER),
            "id": ("代號", 65, tk.CENTER),
            "name": ("名稱", 90, tk.CENTER),
            "action": ("買/賣", 65, tk.CENTER),
            "share": ("股數", 75, tk.E),
            "price": ("成交價", 75, tk.E),
            "reason": ("原因", 110, tk.CENTER),
        }
        for col, (title, width, align) in col_configs.items():
            self.tree.heading(col, text=title)
            self.tree.column(col, width=width, anchor=align)

        # 滾動條
        scroll_y = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll_y.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        # 狀態列
        status_bar = ttk.Frame(self, relief=tk.SUNKEN, padding=(10, 3))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(status_bar, textvariable=self.status_var, foreground="#2C3E50").pack(side=tk.LEFT)

    # ──────────────────────────────────────────
    # 互動事件與功能邏輯
    # ──────────────────────────────────────────
    def _on_stock_id_changed(self, *args):
        sid = self.stock_id_var.get().strip()
        if not sid:
            self.stock_name_var.set("")
            return
        if sid in self.stock_names:
            self.stock_name_var.set(f"✅ {self.stock_names[sid]}")
            self.lbl_stock_name.configure(foreground="#1E8449")
        else:
            self.stock_name_var.set("⚠ 未在名單")
            self.lbl_stock_name.configure(foreground="#E67E22")

    def _on_action_changed(self, *args):
        act = self.action_var.get()
        if act == "buy" and self.reason_var.get() == "REBALANCE":
            self.reason_var.set("ENTRY")
        elif act == "sell" and self.reason_var.get() == "ENTRY":
            self.reason_var.set("REBALANCE")

    def _increment_cycle(self):
        try:
            cur = int(self.cycle_var.get())
            self.cycle_var.set(cur + 1)
            self.status_var.set(f"週期已遞增至 Cycle {cur + 1}")
        except Exception:
            self.cycle_var.set(get_latest_cycle() + 1)

    def _reset_cycle(self):
        self.cycle_var.set(get_latest_cycle())
        self.status_var.set(f"已重置為最新 Cycle {get_latest_cycle()}")

    def _set_today(self):
        self.date_var.set(datetime.now().strftime("%Y/%m/%d"))

    def clear_inputs(self):
        self.stock_id_var.set("")
        self.share_var.set("")
        self.price_var.set("")
        self.id_entry.focus()
        self.status_var.set("已清空輸入欄位")

    def refresh_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        df = get_recent_entries(12)
        if not df.empty:
            for _, row in df.iterrows():
                act_display = "買進" if str(row['action']).lower() == "buy" else "賣出"
                share_str = f"{int(row['share']):,}" if float(row['share']).is_integer() else f"{float(row['share']):.2f}"
                price_str = f"{float(row['price']):.2f}"
                self.tree.insert("", tk.END, values=(
                    int(row['cycle']),
                    row['date'],
                    row['id'],
                    row['name'],
                    act_display,
                    share_str,
                    price_str,
                    row['reason']
                ))

    def submit_entry(self):
        try:
            cycle = int(self.cycle_var.get())
            date = self.date_var.get().strip()
            stock_id = self.stock_id_var.get().strip()
            action = self.action_var.get().strip().lower()
            reason = self.reason_var.get().strip().upper()

            if not stock_id:
                messagebox.showwarning("欄位錯誤", "請輸入股票代號！")
                self.id_entry.focus()
                return

            try:
                share = float(self.share_var.get().strip())
                if share <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showwarning("欄位錯誤", "成交股數必須為大於 0 的數值！")
                return

            try:
                price = float(self.price_var.get().strip())
                if price <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showwarning("欄位錯誤", "成交單價必須為大於 0 的數值！")
                return

            # 送出確認
            name_text = self.stock_names.get(stock_id, "(非白名單)")
            act_text = "買進" if action == "buy" else "賣出"
            confirm_msg = (
                f"請確認交易內容：\n\n"
                f"‧ 週期：Cycle {cycle}\n"
                f"‧ 日期：{date}\n"
                f"‧ 股票：{stock_id} {name_text}\n"
                f"‧ 動作：{act_text} ({action})\n"
                f"‧ 股數：{share:g} 股\n"
                f"‧ 單價：{price:.2f} 元\n"
                f"‧ 原因：{reason}\n\n"
                f"確定要寫入 journal.csv 嗎？"
            )
            if not messagebox.askyesno("寫入確認", confirm_msg):
                return

            success, msg = append_journal_entry(cycle, date, stock_id, action, share, price, reason)
            if success:
                self.status_var.set(msg)
                self.refresh_table()
                self.clear_inputs()
                messagebox.showinfo("成功", f"交易已成功記錄！\n{msg}")
            else:
                messagebox.showerror("失敗", msg)

        except Exception as e:
            messagebox.showerror("例外錯誤", f"處理失敗: {e}")

    def undo_last(self):
        if not messagebox.askyesno("復原確認", "確定要刪除 journal.csv 的『最後一筆』交易紀錄嗎？\n此動作將直接修改檔案！"):
            return

        success, msg = delete_last_entry()
        if success:
            self.status_var.set(msg)
            self.refresh_table()
            messagebox.showinfo("復原成功", msg)
        else:
            messagebox.showerror("復原失敗", msg)


if __name__ == "__main__":
    app = JournalApp()
    app.mainloop()
