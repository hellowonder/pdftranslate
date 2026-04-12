#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import fitz
from PIL import Image, ImageTk

from pdf_crop import (
    canvas_rect_to_pdf_rect,
    pdf_rect_to_canvas_rect,
    remap_selection,
    save_cropped_pdf,
    selection_to_margins,
)


class PdfCropApp:
    EDGE_HIT_WIDTH = 8.0

    def __init__(self, root: tk.Tk, input_path: str | None = None, output_path: str | None = None) -> None:
        self.root = root
        self.root.title("PDF Crop Tool")
        self.doc: fitz.Document | None = None
        self.input_path: str | None = None
        self.default_output_path: str | None = output_path
        self.current_page_index = 0
        self.zoom = 1.3
        self.photo_image: ImageTk.PhotoImage | None = None
        self.shared_selection: fitz.Rect | None = None
        self.shared_selection_page_rect: fitz.Rect | None = None
        self.drag_start: tuple[float, float] | None = None
        self.drag_overlay_id: int | None = None
        self.selection_overlay_id: int | None = None
        self.edge_drag_mode: str | None = None
        self.edge_drag_anchor: fitz.Rect | None = None

        self.page_var = tk.StringVar(value="Page: -/-")
        self.file_var = tk.StringVar(value="No PDF loaded")
        self.status_var = tk.StringVar(value="Drag on the page to select the region to keep.")
        self.margin_vars = {
            "left": tk.StringVar(value="-"),
            "top": tk.StringVar(value="-"),
            "right": tk.StringVar(value="-"),
            "bottom": tk.StringVar(value="-"),
        }

        self._build_ui()
        if input_path:
            self.load_pdf(input_path)

    def _build_ui(self) -> None:
        self.root.geometry("1380x920")
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.grid(row=0, column=0, sticky="ew")
        for idx in range(11):
            toolbar.columnconfigure(idx, weight=0)
        toolbar.columnconfigure(11, weight=1)

        ttk.Button(toolbar, text="Open", command=self.open_dialog).grid(row=0, column=0, padx=4)
        ttk.Button(toolbar, text="Prev", command=self.prev_page).grid(row=0, column=1, padx=4)
        ttk.Button(toolbar, text="Next", command=self.next_page).grid(row=0, column=2, padx=4)
        ttk.Button(toolbar, text="Zoom -", command=lambda: self.adjust_zoom(1 / 1.15)).grid(row=0, column=3, padx=4)
        ttk.Button(toolbar, text="Zoom +", command=lambda: self.adjust_zoom(1.15)).grid(row=0, column=4, padx=4)
        ttk.Button(toolbar, text="Reset", command=self.reset_selection).grid(row=0, column=5, padx=4)
        ttk.Button(toolbar, text="Export PDF", command=self.export_pdf).grid(row=0, column=6, padx=4)
        ttk.Label(toolbar, textvariable=self.page_var).grid(row=0, column=7, padx=12)
        ttk.Label(toolbar, textvariable=self.file_var).grid(row=0, column=11, sticky="e")

        content = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        content.grid(row=1, column=0, sticky="nsew")
        content.rowconfigure(0, weight=1)
        content.columnconfigure(0, weight=1)

        canvas_frame = ttk.Frame(content)
        canvas_frame.grid(row=0, column=0, sticky="nsew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, background="#808080", highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        xscroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        xscroll.grid(row=1, column=0, sticky="ew")
        yscroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)

        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)
        self.canvas.bind("<Motion>", self.on_mouse_move)

        side = ttk.Frame(content, padding=(12, 0, 0, 0))
        side.grid(row=0, column=1, sticky="ns")

        ttk.Label(side, text="Crop Margins (pt)", font=("", 12, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 8))
        for row, key in enumerate(("left", "top", "right", "bottom"), start=1):
            ttk.Label(side, text=key.title()).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Label(side, textvariable=self.margin_vars[key], width=10).grid(row=row, column=1, sticky="e", pady=2)

        ttk.Separator(side, orient="horizontal").grid(row=5, column=0, columnspan=2, sticky="ew", pady=12)
        ttk.Label(side, text="Workflow", font=("", 12, "bold")).grid(row=6, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Label(
            side,
            text=(
                "1. Open a PDF.\n"
                "2. Drag to mark the region to keep.\n"
                "3. Drag a border to fine-tune it.\n"
                "4. Export the cropped PDF."
            ),
            justify="left",
        ).grid(row=7, column=0, columnspan=2, sticky="w")

        ttk.Separator(side, orient="horizontal").grid(row=8, column=0, columnspan=2, sticky="ew", pady=12)
        ttk.Label(side, textvariable=self.status_var, wraplength=240, justify="left").grid(row=9, column=0, columnspan=2, sticky="w")

    def open_dialog(self) -> None:
        path = filedialog.askopenfilename(
            title="Open PDF",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if path:
            self.load_pdf(path)

    def load_pdf(self, path: str) -> None:
        self.close_document()
        self.doc = fitz.open(path)
        self.input_path = os.path.abspath(path)
        self.file_var.set(self.input_path)
        self.current_page_index = 0
        self.shared_selection = None
        self.shared_selection_page_rect = None
        self.status_var.set("Drag on the page to select the region to keep.")
        self.render_page()

    def close_document(self) -> None:
        if self.doc is not None:
            self.doc.close()
            self.doc = None

    def current_page(self) -> fitz.Page:
        if self.doc is None:
            raise RuntimeError("No PDF loaded.")
        return self.doc[self.current_page_index]

    def current_base_rect(self) -> fitz.Rect:
        return fitz.Rect(self.current_page().rect)

    def render_page(self) -> None:
        if self.doc is None:
            self.canvas.delete("all")
            self.page_var.set("Page: -/-")
            self._set_margin_display(None)
            return
        page = self.current_page()
        matrix = fitz.Matrix(self.zoom, self.zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        self.photo_image = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo_image)
        self.canvas.config(scrollregion=(0, 0, pixmap.width, pixmap.height))
        self.page_var.set(f"Page: {self.current_page_index + 1}/{self.doc.page_count}")
        display_selection = self.get_display_selection_for_current_page()
        self._draw_active_selection(display_selection)
        self._set_margin_display(display_selection)

    def _draw_active_selection(self, selection: fitz.Rect | None) -> None:
        if self.selection_overlay_id is not None:
            self.canvas.delete(self.selection_overlay_id)
            self.selection_overlay_id = None
        if selection is None:
            return
        x0, y0, x1, y1 = pdf_rect_to_canvas_rect(selection, self.zoom)
        self.selection_overlay_id = self.canvas.create_rectangle(
            x0,
            y0,
            x1,
            y1,
            outline="#00e676",
            width=2,
        )

    def _set_margin_display(self, selection: fitz.Rect | None) -> None:
        if selection is None:
            for var in self.margin_vars.values():
                var.set("-")
            return
        margins = selection_to_margins(selection, self.current_base_rect())
        self.margin_vars["left"].set(f"{margins.left:.1f}")
        self.margin_vars["top"].set(f"{margins.top:.1f}")
        self.margin_vars["right"].set(f"{margins.right:.1f}")
        self.margin_vars["bottom"].set(f"{margins.bottom:.1f}")

    def on_drag_start(self, event: tk.Event) -> None:
        if self.doc is None:
            return
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        edge_mode = self.hit_test_selection_edge(canvas_x, canvas_y)
        if edge_mode is not None:
            self.edge_drag_mode = edge_mode
            current_selection = self.get_display_selection_for_current_page()
            self.edge_drag_anchor = fitz.Rect(current_selection) if current_selection is not None else None
            return
        self.drag_start = (canvas_x, canvas_y)
        if self.drag_overlay_id is not None:
            self.canvas.delete(self.drag_overlay_id)
            self.drag_overlay_id = None

    def on_drag_move(self, event: tk.Event) -> None:
        x1 = self.canvas.canvasx(event.x)
        y1 = self.canvas.canvasy(event.y)
        if self.edge_drag_mode is not None:
            self.update_selection_by_edge_drag(x1, y1)
            return
        if self.drag_start is None or self.doc is None:
            return
        x0, y0 = self.drag_start
        if self.drag_overlay_id is not None:
            self.canvas.delete(self.drag_overlay_id)
        self.drag_overlay_id = self.canvas.create_rectangle(
            x0,
            y0,
            x1,
            y1,
            outline="#ffb300",
            dash=(6, 4),
            width=2,
        )

    def on_drag_end(self, event: tk.Event) -> None:
        if self.edge_drag_mode is not None:
            self.edge_drag_mode = None
            self.edge_drag_anchor = None
            self.status_var.set("Selection adjusted.")
            return
        if self.drag_start is None or self.doc is None:
            return
        start_x, start_y = self.drag_start
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        self.drag_start = None
        if self.drag_overlay_id is not None:
            self.canvas.delete(self.drag_overlay_id)
            self.drag_overlay_id = None
        try:
            selection = canvas_rect_to_pdf_rect((start_x, start_y, end_x, end_y), self.zoom)
        except ValueError as exc:
            self.status_var.set(str(exc))
            return
        self.shared_selection = selection
        self.shared_selection_page_rect = self.current_base_rect()
        self._draw_active_selection(selection)
        self._set_margin_display(selection)
        self.status_var.set("Shared selection updated.")

    def on_mouse_move(self, event: tk.Event) -> None:
        if self.doc is None:
            return
        if self.edge_drag_mode is not None:
            return
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        edge_mode = self.hit_test_selection_edge(canvas_x, canvas_y)
        self.canvas.configure(cursor=self.cursor_for_edge_mode(edge_mode))

    def prev_page(self) -> None:
        if self.doc is None or self.current_page_index == 0:
            return
        self.current_page_index -= 1
        self.render_page()

    def next_page(self) -> None:
        if self.doc is None or self.current_page_index >= self.doc.page_count - 1:
            return
        self.current_page_index += 1
        self.render_page()

    def adjust_zoom(self, factor: float) -> None:
        if self.doc is None:
            return
        self.zoom = max(0.2, min(self.zoom * factor, 6.0))
        self.render_page()

    def reset_selection(self) -> None:
        if self.doc is None:
            return
        self.shared_selection = None
        self.shared_selection_page_rect = None
        self._draw_active_selection(None)
        self._set_margin_display(None)
        self.status_var.set("Shared crop reset.")

    def export_pdf(self) -> None:
        if self.doc is None or self.input_path is None:
            return
        if self.shared_selection is None or self.shared_selection_page_rect is None:
            messagebox.showerror("No crop selection", "Create a crop selection before exporting.")
            return
        initial_path = self.default_output_path or self._default_output_path(self.input_path)
        output_path = filedialog.asksaveasfilename(
            title="Export Cropped PDF",
            defaultextension=".pdf",
            initialfile=Path(initial_path).name,
            initialdir=str(Path(initial_path).parent),
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if not output_path:
            return
        page_selections = {
            page_index: remap_selection(
                self.shared_selection,
                self.shared_selection_page_rect,
                fitz.Rect(self.doc[page_index].rect),
            )
            for page_index in range(self.doc.page_count)
        }
        try:
            save_cropped_pdf(self.input_path, output_path, page_selections)
        except Exception as exc:  # pragma: no cover - GUI error path
            messagebox.showerror("Export failed", str(exc))
            return
        self.status_var.set(f"Exported cropped PDF to {output_path}")
        self.default_output_path = output_path

    def get_display_selection_for_current_page(self) -> fitz.Rect | None:
        if self.shared_selection is None or self.shared_selection_page_rect is None or self.doc is None:
            return None
        return remap_selection(self.shared_selection, self.shared_selection_page_rect, self.current_base_rect())

    def hit_test_selection_edge(self, canvas_x: float, canvas_y: float) -> str | None:
        selection = self.get_display_selection_for_current_page()
        if selection is None:
            return None
        x0, y0, x1, y1 = pdf_rect_to_canvas_rect(selection, self.zoom)
        if not (x0 - self.EDGE_HIT_WIDTH <= canvas_x <= x1 + self.EDGE_HIT_WIDTH):
            return None
        if not (y0 - self.EDGE_HIT_WIDTH <= canvas_y <= y1 + self.EDGE_HIT_WIDTH):
            return None
        if abs(canvas_x - x0) <= self.EDGE_HIT_WIDTH and y0 <= canvas_y <= y1:
            return "left"
        if abs(canvas_x - x1) <= self.EDGE_HIT_WIDTH and y0 <= canvas_y <= y1:
            return "right"
        if abs(canvas_y - y0) <= self.EDGE_HIT_WIDTH and x0 <= canvas_x <= x1:
            return "top"
        if abs(canvas_y - y1) <= self.EDGE_HIT_WIDTH and x0 <= canvas_x <= x1:
            return "bottom"
        return None

    def update_selection_by_edge_drag(self, canvas_x: float, canvas_y: float) -> None:
        if self.edge_drag_mode is None or self.edge_drag_anchor is None:
            return
        anchor = fitz.Rect(self.edge_drag_anchor)
        pdf_x = canvas_x / self.zoom
        pdf_y = canvas_y / self.zoom
        if self.edge_drag_mode == "left":
            anchor.x0 = pdf_x
        elif self.edge_drag_mode == "right":
            anchor.x1 = pdf_x
        elif self.edge_drag_mode == "top":
            anchor.y0 = pdf_y
        elif self.edge_drag_mode == "bottom":
            anchor.y1 = pdf_y
        try:
            adjusted = remap_selection(anchor, self.current_base_rect(), self.current_base_rect())
        except ValueError:
            return
        self.shared_selection = adjusted
        self.shared_selection_page_rect = self.current_base_rect()
        self._draw_active_selection(adjusted)
        self._set_margin_display(adjusted)

    @staticmethod
    def cursor_for_edge_mode(edge_mode: str | None) -> str:
        if edge_mode in {"left", "right"}:
            return "sb_h_double_arrow"
        if edge_mode in {"top", "bottom"}:
            return "sb_v_double_arrow"
        return "crosshair"

    @staticmethod
    def _default_output_path(input_path: str) -> str:
        source = Path(input_path)
        return str(source.with_name(f"{source.stem}.cropped.pdf"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive PDF crop box editor.")
    parser.add_argument("--input", help="Optional input PDF path to open at launch.")
    parser.add_argument("--output", help="Optional default output PDF path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    app = PdfCropApp(root, input_path=args.input, output_path=args.output)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.close_document(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
