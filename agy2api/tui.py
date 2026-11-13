import sys
from typing import List, Optional, Tuple


def fuzzy_score(query: str, target: str) -> int:
    """Calculate a fuzzy match score between query and target string. Higher is better."""
    q = query.strip().lower()
    t = target.strip().lower()
    if not q:
        return 1
    if q == t:
        return 1000
    if t.startswith(q):
        return 500 - len(t)
    if q in t:
        return 300 - len(t)

    tokens = q.split()
    if len(tokens) > 1 and all(tok in t for tok in tokens):
        return 200 - len(t)

    # Subsequence match
    t_idx = 0
    matches = 0
    for char in q:
        while t_idx < len(t) and t[t_idx] != char:
            t_idx += 1
        if t_idx < len(t):
            matches += 1
            t_idx += 1
        else:
            break

    if matches == len(q):
        return 100 - len(t)

    return 0


def get_key() -> str:
    """Read a single keypress cross-platform."""
    if sys.platform == "win32":
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            ch2 = msvcrt.getwch()
            if ch2 == "H":
                return "UP"
            elif ch2 == "P":
                return "DOWN"
            elif ch2 == "K":
                return "LEFT"
            elif ch2 == "M":
                return "RIGHT"
            elif ch2 == "S":
                return "DELETE"
            return ""
        elif ch == "\r":
            return "ENTER"
        elif ch in ("\x08", "\x7f"):
            return "BACKSPACE"
        elif ch == "\x1b":
            return "ESC"
        elif ch == "\t":
            return "TAB"
        elif ch == "\x13":  # Ctrl+S
            return "SAVE"
        elif ch == "\x03":  # Ctrl+C
            return "QUIT"
        return ch
    else:
        import termios
        import tty
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                # Check for escape sequence
                seq = sys.stdin.read(2)
                if seq == "[A":
                    return "UP"
                elif seq == "[B":
                    return "DOWN"
                elif seq == "[C":
                    return "RIGHT"
                elif seq == "[D":
                    return "LEFT"
                elif seq == "[3":
                    sys.stdin.read(1)  # trailing ~
                    return "DELETE"
                return "ESC"
            elif ch in ("\r", "\n"):
                return "ENTER"
            elif ch in ("\x7f", "\x08"):
                return "BACKSPACE"
            elif ch == "\t":
                return "TAB"
            elif ch == "\x13":  # Ctrl+S
                return "SAVE"
            elif ch == "\x03":  # Ctrl+C
                return "QUIT"
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def run_model_manager_tui(
    upstream_models: List[str],
    initial_selected: Optional[List[str]] = None,
    default_model: str = "",
) -> List[str]:
    """Launch the interactive 32-Model Manager Terminal UI."""
    selected: List[str] = list(initial_selected or [])

    # Check if stdout/stdin is interactive
    if not sys.stdout.isatty() or not sys.stdin.isatty():
        return selected

    query = ""
    match_index = 0
    selected_cursor = 0
    active_pane = "search"  # "search" or "selected"
    status_msg = "Type to search models. [Enter] to add. [Tab] to manage list. [Esc] to save."

    def clear_screen():
        # Clear screen and move cursor to home
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

    while True:
        # 1. Filter matches
        scored: List[Tuple[int, str]] = []
        for m in upstream_models:
            score = fuzzy_score(query, m)
            if score > 0:
                scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        matches = [m for _, m in scored[:10]]

        if match_index >= len(matches):
            match_index = max(0, len(matches) - 1)
        if selected and selected_cursor >= len(selected):
            selected_cursor = max(0, len(selected) - 1)

        # 2. Render UI
        clear_screen()
        lines: List[str] = []
        lines.append("\033[1;36m=== agy2api Model Manager ===\033[0m")
        count_color = "\033[32m" if len(selected) < 32 else "\033[33m"
        lines.append(f"Active Slots: {count_color}{len(selected)}/32\033[0m  |  Mode: \033[1m{active_pane.upper()}\033[0m")
        lines.append("\033[90m" + "─" * 60 + "\033[0m")

        # Section A: Selected Models
        lines.append(f"\033[1;33mSelected Models ({len(selected)}/32):\033[0m")
        if not selected:
            lines.append("  \033[90m(None selected yet. Search below and press Enter to add)\033[0m")
        else:
            # Display in 2 columns if more than 8
            for i, sm in enumerate(selected):
                prefix = "\033[1;31m> \033[0m" if (active_pane == "selected" and i == selected_cursor) else "  "
                del_hint = " \033[90m[Del to remove]\033[0m" if (active_pane == "selected" and i == selected_cursor) else ""
                lines.append(f"{prefix}\033[32m{i+1:2d}.\033[0m {sm}{del_hint}")

        lines.append("\033[90m" + "─" * 60 + "\033[0m")

        # Section B: Search & Live Matches
        search_indicator = "\033[1;32m>\033[0m " if active_pane == "search" else "  "
        lines.append(f"{search_indicator}\033[1mSearch Upstream:\033[0m {query}\033[7m \033[0m")

        if matches:
            lines.append("\033[90mMatching Models (Top 10):\033[0m")
            for i, m in enumerate(matches):
                is_selected = m in selected
                is_curr = (active_pane == "search" and i == match_index)

                if is_curr:
                    pointer = "\033[1;36m▶ \033[0m"
                    fmt = "\033[1;37;44m"  # highlighted background
                    end_fmt = "\033[0m"
                else:
                    pointer = "  "
                    fmt = ""
                    end_fmt = ""

                badge = " \033[32m[Added]\033[0m" if is_selected else ""
                lines.append(f"{pointer}{fmt} {m} {end_fmt}{badge}")
        else:
            lines.append("  \033[90mNo matching models found.\033[0m")

        lines.append("\033[90m" + "─" * 60 + "\033[0m")
        lines.append(f"\033[94m{status_msg}\033[0m")
        lines.append("\033[90mControls: [Tab] Switch pane | [↑/↓] Navigate | [Enter] Add | [Del] Remove | [Esc] Save & Exit\033[0m")

        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

        # 3. Handle Key
        key = get_key()

        if key in ("ESC", "SAVE"):
            break
        elif key == "QUIT":
            break
        elif key == "TAB":
            active_pane = "selected" if active_pane == "search" else "search"
            status_msg = "Managing selected models. Press [Del] or [Backspace] to remove." if active_pane == "selected" else "Searching models. Press [Enter] to add."
        elif key == "UP":
            if active_pane == "search":
                if match_index > 0:
                    match_index -= 1
            else:
                if selected_cursor > 0:
                    selected_cursor -= 1
        elif key == "DOWN":
            if active_pane == "search":
                if match_index < len(matches) - 1:
                    match_index += 1
            else:
                if selected_cursor < len(selected) - 1:
                    selected_cursor += 1
        elif key == "ENTER":
            if active_pane == "search":
                if matches:
                    chosen = matches[match_index]
                    if chosen in selected:
                        selected.remove(chosen)
                        status_msg = f"Removed '{chosen}'."
                    else:
                        if len(selected) >= 32:
                            status_msg = "⚠️ Maximum 32 models reached! Remove one before adding."
                        else:
                            selected.append(chosen)
                            status_msg = f"Added '{chosen}' ({len(selected)}/32)."
            else:
                # Toggle back to search
                active_pane = "search"
        elif key in ("BACKSPACE", "DELETE"):
            if active_pane == "search":
                if query:
                    query = query[:-1]
                    match_index = 0
            else:
                if selected:
                    removed = selected.pop(selected_cursor)
                    status_msg = f"Removed '{removed}'."
                    if selected_cursor >= len(selected):
                        selected_cursor = max(0, len(selected) - 1)
        else:
            if active_pane == "search" and len(key) == 1 and key.isprintable():
                query += key
                match_index = 0

    clear_screen()
    print(f"\n[OK] Model selection updated ({len(selected)} models configured).")
    return selected
