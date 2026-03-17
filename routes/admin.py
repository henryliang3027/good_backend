from nicegui import ui
from dependencies import get_collection


def _upsert(collection, new_id: str, brand: str, flavor: str, color: str, count: int, old_id: str | None = None):
    if old_id and old_id != new_id:
        collection.delete(ids=[old_id])
    collection.upsert(
        ids=[new_id],
        embeddings=[[0.0]],  # placeholder；此頁面不做 embedding 比對
        metadatas=[{"brand": brand, "flavor": flavor, "color": color, "count": count}],
    )


def _fetch_rows(collection) -> list[dict]:
    result = collection.get()
    return [
        {
            "id": iid,
            "brand": m.get("brand", ""),
            "flavor": m.get("flavor", ""),
            "color": m.get("color", ""),
            "count": int(m.get("count", 0)),
        }
        for iid, m in zip(result["ids"], result["metadatas"])
    ]


def _open_form_dialog(collection, table, total_label, row: dict | None = None):
    """新增 / 編輯 dialog。row=None 時為新增模式。"""
    is_edit = row is not None

    with ui.dialog().props("persistent") as dialog, ui.card().classes("w-96 gap-2"):
        ui.label("編輯飲料" if is_edit else "新增飲料").classes("text-lg font-bold")

        brand_in  = ui.input("品牌 *",  value=row["brand"]  if is_edit else "").classes("w-full")
        flavor_in = ui.input("口味 *",  value=row["flavor"] if is_edit else "").classes("w-full")
        color_in  = ui.input("顏色",    value=row["color"]  if is_edit else "").classes("w-full")
        count_in  = ui.number("瓶數",   value=row["count"]  if is_edit else 0, min=0, precision=0).classes("w-full")

        def on_save():
            brand  = brand_in.value.strip()
            flavor = flavor_in.value.strip()
            color  = color_in.value.strip()
            count  = int(count_in.value or 0)

            if not brand or not flavor:
                ui.notify("品牌和口味為必填", color="negative", position="top")
                return

            new_id = f"{brand}{flavor}"
            old_id = row["id"] if is_edit else None
            _upsert(collection, new_id, brand, flavor, color, count, old_id)

            table.rows[:] = _fetch_rows(collection)
            table.update()
            total_label.set_text(f"共 {len(table.rows)} 種商品")
            ui.notify("儲存成功 ✓", color="positive", position="top")
            dialog.close()

        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            ui.button("取消", on_click=dialog.close).props("flat")
            ui.button("儲存", on_click=on_save).props("color=primary")

    dialog.open()


def _open_delete_dialog(collection, table, total_label, row: dict):
    with ui.dialog().props("persistent") as dialog, ui.card().classes("w-80 gap-4"):
        ui.label(f'確定要刪除「{row["id"]}」？').classes("text-base")

        def do_delete():
            collection.delete(ids=[row["id"]])
            table.rows[:] = [r for r in table.rows if r["id"] != row["id"]]
            table.update()
            total_label.set_text(f"共 {len(table.rows)} 種商品")
            ui.notify(f'已刪除：{row["id"]}', color="warning", position="top")
            dialog.close()

        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("取消", on_click=dialog.close).props("flat")
            ui.button("確定刪除", on_click=do_delete).props("color=negative")

    dialog.open()


@ui.page("/admin")
def admin_page():
    collection = get_collection()


    # ── Header ──────────────────────────────────────────────────────────
    with ui.row().classes("w-full items-center justify-between mb-2"):
        ui.label("飲料庫存管理").classes("text-2xl font-bold")
        ui.button("＋ 新增飲料",
                  on_click=lambda: _open_form_dialog(collection, table, total_label)
                  ).props("color=primary")

    rows = _fetch_rows(collection)
    total_label = ui.label(f"共 {len(rows)} 種商品").classes("text-gray-500 text-sm mb-2")

    # ── Table ────────────────────────────────────────────────────────────
    columns = [
        {"name": "brand",   "label": "品牌", "field": "brand",  "align": "left",   "sortable": True},
        {"name": "flavor",  "label": "口味", "field": "flavor", "align": "left",   "sortable": True},
        {"name": "color",   "label": "顏色", "field": "color",  "align": "left"},
        {"name": "count",   "label": "瓶數", "field": "count",  "align": "center", "sortable": True},
        {"name": "actions", "label": "操作", "field": "id",     "align": "center"},
    ]

    table = ui.table(columns=columns, rows=rows, row_key="id").classes("w-full")

    # 操作欄：編輯 / 刪除 icon button
    table.add_slot("body-cell-actions", """
        <q-td :props="props" auto-width>
            <q-btn flat dense round color="primary" icon="edit"
                   @click="$parent.$emit('edit', props.row)" />
            <q-btn flat dense round color="negative" icon="delete"
                   @click="$parent.$emit('delete', props.row)" />
        </q-td>
    """)

    table.on("edit",   lambda e: _open_form_dialog(collection, table, total_label, e.args))
    table.on("delete", lambda e: _open_delete_dialog(collection, table, total_label, e.args))
