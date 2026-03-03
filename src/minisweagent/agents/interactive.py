"""A small generalization of the default agent that puts the user in the loop.

There are three modes:
- human: commands issued by the user are executed immediately
- confirm: commands issued by the LM but not whitelisted are confirmed by the user
- yolo: commands issued by the LM are executed immediately without confirmation
"""

import re
from typing import Literal, NoReturn

from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import PromptSession
from rich.console import Console
from rich.rule import Rule

from minisweagent import global_config_dir
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import LimitsExceeded, Submitted, UserInterruption
from minisweagent.models.utils.content_string import get_content_string

console = Console(highlight=False)
_history = FileHistory(global_config_dir / "interactive_history.txt")
_prompt_session = PromptSession(history=_history)
_multiline_prompt_session = PromptSession(history=_history, multiline=True)

_CANDIDATE_STYLE = "bold cyan"
_VERIFIER_STYLE = "bold magenta"
_FINAL_ACTION_STYLE = "bold green"


class InteractiveAgentConfig(AgentConfig):
    mode: Literal["human", "confirm", "yolo"] = "confirm"
    """Whether to confirm actions."""
    whitelist_actions: list[str] = []
    """Never confirm actions that match these regular expressions."""
    confirm_exit: bool = True
    """If the agent wants to finish, do we ask for confirmation from user?"""
    show_all_candidate_actions: bool = False
    """Show all sampled candidate actions before execution."""
    show_verifier_summary_output: bool = False
    """Print concise verifier output summary (scores + checklist info)."""
    show_full_verifier_output: bool = False
    """Print the full verifier output payload for debugging."""


def _multiline_prompt() -> str:
    return _multiline_prompt_session.prompt(
        "",
        bottom_toolbar=HTML(
            "Submit message: <b fg='yellow' bg='black'>Esc, then Enter</b> | "
            "Navigate history: <b fg='yellow' bg='black'>Arrow Up/Down</b> | "
            "Search history: <b fg='yellow' bg='black'>Ctrl+R</b>"
        ),
    )


class InteractiveAgent(DefaultAgent):
    _MODE_COMMANDS_MAPPING = {"/u": "human", "/c": "confirm", "/y": "yolo"}

    def __init__(self, *args, config_class=InteractiveAgentConfig, **kwargs):
        super().__init__(*args, config_class=config_class, **kwargs)
        self.cost_last_confirmed = 0.0

    def add_messages(self, *messages: dict) -> list[dict]:
        # Extend supermethod to print messages
        for msg in messages:
            role, content = msg.get("role") or msg.get("type", "unknown"), get_content_string(msg)
            if role == "assistant":
                console.print(
                    f"\n[red][bold]mini-swe-agent[/bold] (step [bold]{self.step_count + 1}[/bold], [bold]${self.cost:.2f}[/bold]):[/red]\n",
                    end="",
                    highlight=False,
                )
                self._print_verifier_candidate_scores(msg)
                self._print_all_candidate_actions(msg)
                self._print_verifier_summary_output(msg)
                self._print_full_verifier_output(msg)
                console.print("Final action:", highlight=False, markup=False, style=_FINAL_ACTION_STYLE)
            else:
                console.print(f"\n[bold green]{role.capitalize()}[/bold green]:\n", end="", highlight=False)
            console.print(content, highlight=False, markup=False)
        return super().add_messages(*messages)

    def _print_verifier_candidate_scores(self, message: dict) -> None:
        extra = message.get("extra", {}) or {}
        verifier = extra.get("verifier", {}) or {}
        if not isinstance(verifier, dict) or not verifier.get("enabled"):
            return

        candidates = verifier.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return

        selection_index_base = verifier.get("selection_index_base", 1)
        selected_index = verifier.get("selected_index")
        rewards = self._extract_reward_scores(verifier)
        verifier_type = verifier.get("type", "unknown")
        console.print(f"Candidate actions ({verifier_type}):", highlight=False, markup=False, style=_CANDIDATE_STYLE)

        for i, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            raw_index = candidate.get("index", i)
            index = raw_index if isinstance(raw_index, int) else i
            display_index = index + selection_index_base if isinstance(selection_index_base, int) else index + 1
            score = rewards[index] if index < len(rewards) else None
            score_text = self._format_score(score)
            selected_prefix = "*" if selected_index == index else " "
            commands = self._candidate_commands(candidate)
            command_text = self._truncate_inline(" ; ".join(commands) if commands else "<no parsed action>")
            thought_text = self._candidate_thought(candidate) or "<none>"
            console.print(f"{selected_prefix} Candidate {display_index}", highlight=False, markup=False)
            console.print(
                f"  {command_text} ({score_text})",
                highlight=False,
                markup=False,
            )
            console.print(
                f"  {thought_text}",
                highlight=False,
                markup=False,
            )

    def _print_all_candidate_actions(self, message: dict) -> None:
        if not self.config.show_all_candidate_actions:
            return

        verifier = self._get_verifier_metadata(message)
        if verifier is None:
            return
        # If verifier score table already printed, actions are already shown there.
        if verifier.get("enabled"):
            return

        candidates = verifier.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return

        selection_index_base = verifier.get("selection_index_base", 1)
        selected_index = verifier.get("selected_index")
        rewards = self._extract_reward_scores(verifier)
        verifier_type = verifier.get("type", "none")
        console.print(f"Candidate actions (type={verifier_type}):", highlight=False, markup=False, style=_CANDIDATE_STYLE)
        for i, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            raw_index = candidate.get("index", i)
            index = raw_index if isinstance(raw_index, int) else i
            display_index = index + selection_index_base if isinstance(selection_index_base, int) else index + 1
            commands = self._candidate_commands(candidate)
            command_text = self._truncate_inline(" ; ".join(commands) if commands else "<no parsed action>")
            thought_text = self._candidate_thought(candidate) or "<none>"
            score = rewards[index] if index < len(rewards) else None
            selected_prefix = "*" if selected_index == index else " "
            console.print(f"{selected_prefix} Candidate {display_index}", highlight=False, markup=False)
            console.print(
                f"  {command_text} ({self._format_score(score)})",
                highlight=False,
                markup=False,
            )
            console.print(
                f"  {thought_text}",
                highlight=False,
                markup=False,
            )

    def _print_verifier_summary_output(self, message: dict) -> None:
        if not self.config.show_verifier_summary_output:
            return

        verifier = self._get_verifier_metadata(message)
        if verifier is None or not verifier.get("enabled"):
            return

        verifier_output = verifier.get("verifier_output")
        if not isinstance(verifier_output, dict):
            return

        selection_index_base = verifier.get("selection_index_base", 1)
        selected_index = verifier.get("selected_index")
        selected_display = selected_index + selection_index_base if isinstance(selected_index, int) else "n/a"
        verifier_type = verifier.get("type", "unknown")
        console.print(
            f"Verifier summary ({verifier_type}): selected=C{selected_display}",
            highlight=False,
            markup=False,
            style=_VERIFIER_STYLE,
        )

        scores = self._extract_reward_scores(verifier)
        if scores:
            console.print(
                f"  scores: {self._format_candidate_scores(scores, selection_index_base)}",
                highlight=False,
                markup=False,
            )

        progress_scores = verifier_output.get("candidate_progress_scores")
        if isinstance(progress_scores, list) and progress_scores:
            console.print(
                f"  progress: {self._format_candidate_scores(progress_scores, selection_index_base)}",
                highlight=False,
                markup=False,
            )
        elif isinstance(verifier_output.get("progress_score"), (int, float)):
            console.print(
                f"  progress: {self._format_score(verifier_output.get('progress_score'))}",
                highlight=False,
                markup=False,
            )

        checklist = verifier_output.get("checklist")
        if isinstance(checklist, dict):
            checklist_items = checklist.get("items")
            checklist_count = len(checklist_items) if isinstance(checklist_items, list) else 0
            parts = [f"items={checklist_count}"]
            if isinstance(checklist.get("dynamic"), bool):
                parts.append(f"dynamic={checklist.get('dynamic')}")
            if isinstance(checklist.get("update_mode"), str) and checklist.get("update_mode"):
                parts.append(f"update_mode={checklist.get('update_mode')}")
            console.print(f"  checklist: {', '.join(parts)}", highlight=False, markup=False)

        checklist_item_scores = verifier_output.get("checklist_item_scores")
        if isinstance(checklist_item_scores, list) and checklist_item_scores:
            console.print(
                f"  checklist_scores: {self._format_item_scores(checklist_item_scores)}",
                highlight=False,
                markup=False,
            )

        candidate_checklist_scores = verifier_output.get("candidate_checklist_item_scores")
        if (
            isinstance(candidate_checklist_scores, list)
            and isinstance(selected_index, int)
            and 0 <= selected_index < len(candidate_checklist_scores)
        ):
            selected_candidate_scores = candidate_checklist_scores[selected_index]
            if isinstance(selected_candidate_scores, list) and selected_candidate_scores:
                console.print(
                    f"  checklist_scores(selected): {self._format_item_scores(selected_candidate_scores)}",
                    highlight=False,
                    markup=False,
                )

    def _print_full_verifier_output(self, message: dict) -> None:
        if not self.config.show_full_verifier_output:
            return

        verifier = self._get_verifier_metadata(message)
        if verifier is None:
            return

        verifier_output = verifier.get("verifier_output")
        verifier_enabled = bool(verifier.get("enabled"))
        if verifier_output in (None, {}) and not verifier_enabled:
            return

        content_output = self._extract_verifier_content_output(verifier_output, verifier)
        if not content_output:
            return

        verifier_type = verifier.get("type", "unknown")
        console.print(f"Verifier output ({verifier_type}):", highlight=False, markup=False, style=_VERIFIER_STYLE)
        console.print(content_output, highlight=False, markup=False)

    def _get_verifier_metadata(self, message: dict) -> dict | None:
        extra = message.get("extra", {}) or {}
        verifier = extra.get("verifier", {}) or {}
        if isinstance(verifier, dict):
            return verifier
        return None

    def _extract_reward_scores(self, verifier: dict) -> list[float | None]:
        verifier_output = verifier.get("verifier_output", {}) or {}
        if not isinstance(verifier_output, dict):
            return []
        rewards = verifier_output.get("rewards")
        if not isinstance(rewards, list):
            rewards = verifier_output.get("scores")
        if not isinstance(rewards, list):
            return []
        parsed_rewards: list[float | None] = []
        for reward in rewards:
            if isinstance(reward, (int, float)):
                parsed_rewards.append(float(reward))
            else:
                parsed_rewards.append(None)
        return parsed_rewards

    def _candidate_commands(self, candidate: dict) -> list[str]:
        commands: list[str] = []
        actions = candidate.get("actions")
        if isinstance(actions, list):
            for action in actions:
                if not isinstance(action, dict):
                    continue
                command = action.get("command")
                if isinstance(command, str) and command:
                    commands.append(command)
        if commands:
            return commands
        action = candidate.get("action")
        if isinstance(action, str) and action:
            return [action]
        return []

    def _candidate_thought(self, candidate: dict) -> str | None:
        paired_thought = candidate.get("paired_thought")
        if isinstance(paired_thought, str):
            normalized = paired_thought.strip()
            if normalized:
                return normalized

        content = candidate.get("content")
        if not isinstance(content, str):
            return None
        normalized = content.strip()
        if not normalized:
            return None
        # Keep only the thought text and drop command blocks from mixed content.
        if re.match(r"^\s*THOUGHTS?:", normalized, flags=re.IGNORECASE):
            normalized = normalized.split(":", 1)[1].strip()
        if "```" in normalized:
            normalized = normalized.split("```", 1)[0].strip()
        return normalized or None

    def _format_score(self, score: float | int | None) -> str:
        if isinstance(score, (int, float)):
            return f"{float(score):.4f}"
        return "n/a"

    def _format_candidate_scores(self, scores: list[float | int | None], selection_index_base: int | None) -> str:
        base = selection_index_base if isinstance(selection_index_base, int) else 1
        parts: list[str] = []
        for index, score in enumerate(scores):
            parts.append(f"C{index + base}={self._format_score(score)}")
        return ", ".join(parts)

    def _format_item_scores(self, scores: list[float | int | None]) -> str:
        parts: list[str] = []
        for index, score in enumerate(scores):
            parts.append(f"I{index + 1}={self._format_score(score)}")
        return ", ".join(parts)

    def _truncate_inline(self, text: str, max_chars: int = 100) -> str:
        single_line = " ".join(text.split())
        if len(single_line) <= max_chars:
            return single_line
        if max_chars <= 3:
            return single_line[:max_chars]
        return f"{single_line[: max_chars - 3]}..."

    def _extract_verifier_content_output(self, verifier_output: object, verifier: dict) -> str:
        if isinstance(verifier_output, str):
            return verifier_output
        if not isinstance(verifier_output, dict):
            return ""

        raw_output = verifier_output.get("raw_output")
        if isinstance(raw_output, str):
            return raw_output

        rewards = verifier_output.get("rewards")
        reward_values = rewards if isinstance(rewards, list) else []
        raw_outputs = verifier_output.get("raw_outputs")
        if not isinstance(raw_outputs, list):
            return self._format_reward_outputs_only(reward_values, verifier)

        selection_index_base = verifier.get("selection_index_base", 1)
        base = selection_index_base if isinstance(selection_index_base, int) else 1
        selected_index = verifier.get("selected_index")
        rendered_outputs: list[str] = []
        for index, output in enumerate(raw_outputs):
            if not isinstance(output, str):
                continue
            header_parts: list[str] = []
            if isinstance(selected_index, int) and selected_index == index:
                header_parts.append("selected")
            if reward_values:
                reward = reward_values[index] if index < len(reward_values) else None
                header_parts.append(f"reward={self._format_score(reward)}")
            suffix = f" ({', '.join(header_parts)})" if header_parts else ""
            rendered_outputs.append(f"Candidate {index + base}{suffix}:\n{output}")
        if rendered_outputs:
            return "\n\n".join(rendered_outputs)
        return self._format_reward_outputs_only(reward_values, verifier)

    def _format_reward_outputs_only(self, rewards: list[object], verifier: dict) -> str:
        if not rewards:
            return ""
        selection_index_base = verifier.get("selection_index_base", 1)
        base = selection_index_base if isinstance(selection_index_base, int) else 1
        selected_index = verifier.get("selected_index")
        lines: list[str] = []
        for index, reward in enumerate(rewards):
            header_parts: list[str] = []
            if isinstance(selected_index, int) and selected_index == index:
                header_parts.append("selected")
            header_parts.append(f"reward={self._format_score(reward if isinstance(reward, (int, float)) else None)}")
            lines.append(f"Candidate {index + base} ({', '.join(header_parts)})")
        return "\n".join(lines)

    def query(self) -> dict:
        # Extend supermethod to handle human mode
        if self.config.mode == "human":
            match command := self._prompt_and_handle_slash_commands("[bold yellow]>[/bold yellow] "):
                case "/y" | "/c":
                    pass
                case _:
                    msg = {
                        "role": "user",
                        "content": f"User command: \n```bash\n{command}\n```",
                        "extra": {"actions": [{"command": command}]},
                    }
                    self.add_messages(msg)
                    return msg
        try:
            #with console.status("Waiting for the LM to respond..."):
            return super().query()
        except LimitsExceeded:
            console.print(
                f"Limits exceeded. Limits: {self.config.step_limit} steps, ${self.config.cost_limit}.\n"
                f"Current spend: {self.step_count} steps, ${self.cost:.2f}."
            )
            self.config.step_limit = int(input("New step limit: "))
            self.config.cost_limit = float(input("New cost limit: "))
            return super().query()

    def step(self) -> list[dict]:
        # Override the step method to handle user interruption
        try:
            console.print(Rule())
            return super().step()
        except KeyboardInterrupt:
            interruption_message = self._prompt_and_handle_slash_commands(
                "\n\n[bold yellow]Interrupted.[/bold yellow] "
                "[green]Type a comment/command[/green] (/h for available commands)"
                "\n[bold yellow]>[/bold yellow] "
            ).strip()
            if not interruption_message or interruption_message in self._MODE_COMMANDS_MAPPING:
                interruption_message = "Temporary interruption caught."
            raise UserInterruption(
                {
                    "role": "user",
                    "content": f"Interrupted by user: {interruption_message}",
                    "extra": {"interrupt_type": "UserInterruption"},
                }
            )

    def execute_actions(self, message: dict) -> list[dict]:
        # Override to handle user confirmation and confirm_exit, with try/finally to preserve partial outputs
        actions = message.get("extra", {}).get("actions", [])
        commands = [action["command"] for action in actions]
        outputs = []
        try:
            self._ask_confirmation_or_interrupt(commands)
            for action in actions:
                outputs.append(self.env.execute(action))
        except Submitted as e:
            self._check_for_new_task_or_submit(e)
        finally:
            result = self.add_messages(
                *self.model.format_observation_messages(message, outputs, self.get_template_vars())
            )
        return result

    def _add_observation_messages(self, message: dict, outputs: list[dict]) -> list[dict]:
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def _check_for_new_task_or_submit(self, e: Submitted) -> NoReturn:
        """Check if user wants to add a new task or submit."""
        if self.config.confirm_exit:
            message = (
                "[bold yellow]Agent wants to finish.[/bold yellow] "
                "[bold green]Type new task[/bold green] or [red][bold]Esc, then enter[/bold] to quit.[/red]\n"
                "[bold yellow]>[/bold yellow] "
            )
            if new_task := self._prompt_and_handle_slash_commands(message, _multiline=True).strip():
                raise UserInterruption(
                    {
                        "role": "user",
                        "content": f"The user added a new task: {new_task}",
                        "extra": {"interrupt_type": "UserNewTask"},
                    }
                )
        raise e

    def _should_ask_confirmation(self, action: str) -> bool:
        return self.config.mode == "confirm" and not any(re.match(r, action) for r in self.config.whitelist_actions)

    def _ask_confirmation_or_interrupt(self, commands: list[str]) -> None:
        commands_needing_confirmation = [c for c in commands if self._should_ask_confirmation(c)]
        if not commands_needing_confirmation:
            return
        n = len(commands_needing_confirmation)
        prompt = (
            f"[bold yellow]Execute {n} action(s)?[/] [green][bold]Enter[/] to confirm[/], "
            "[red]type [bold]comment[/] to reject[/], or [blue][bold]/h[/] to show available commands[/]\n"
            "[bold yellow]>[/bold yellow] "
        )
        match user_input := self._prompt_and_handle_slash_commands(prompt).strip():
            case "" | "/y":
                pass  # confirmed, do nothing
            case "/u":  # Skip execution action and get back to query
                raise UserInterruption(
                    {
                        "role": "user",
                        "content": "Commands not executed. Switching to human mode",
                        "extra": {"interrupt_type": "UserRejection"},
                    }
                )
            case _:
                raise UserInterruption(
                    {
                        "role": "user",
                        "content": f"Commands not executed. The user rejected your commands with the following message: {user_input}",
                        "extra": {"interrupt_type": "UserRejection"},
                    }
                )

    def _prompt_and_handle_slash_commands(self, prompt: str, *, _multiline: bool = False) -> str:
        """Prompts the user, takes care of /h (followed by requery) and sets the mode. Returns the user input."""
        console.print(prompt, end="")
        if _multiline:
            return _multiline_prompt()
        user_input = _prompt_session.prompt("")
        if user_input == "/m":
            return self._prompt_and_handle_slash_commands(prompt, _multiline=True)
        if user_input == "/h":
            console.print(
                f"Current mode: [bold green]{self.config.mode}[/bold green]\n"
                f"[bold green]/y[/bold green] to switch to [bold yellow]yolo[/bold yellow] mode (execute LM commands without confirmation)\n"
                f"[bold green]/c[/bold green] to switch to [bold yellow]confirmation[/bold yellow] mode (ask for confirmation before executing LM commands)\n"
                f"[bold green]/u[/bold green] to switch to [bold yellow]human[/bold yellow] mode (execute commands issued by the user)\n"
                f"[bold green]/m[/bold green] to enter multiline comment",
            )
            return self._prompt_and_handle_slash_commands(prompt)
        if user_input in self._MODE_COMMANDS_MAPPING:
            if self.config.mode == self._MODE_COMMANDS_MAPPING[user_input]:
                return self._prompt_and_handle_slash_commands(
                    f"[bold red]Already in {self.config.mode} mode.[/bold red]\n{prompt}"
                )
            self.config.mode = self._MODE_COMMANDS_MAPPING[user_input]
            console.print(f"Switched to [bold green]{self.config.mode}[/bold green] mode.")
            return user_input
        return user_input
