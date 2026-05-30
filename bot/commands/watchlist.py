# -*- coding: utf-8 -*-
"""Manage the GitHub Actions STOCK_LIST variable from chat."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

import requests

from bot.commands.base import BotCommand
from bot.models import BotMessage, BotResponse
from src.data.stock_mapping import STOCK_NAME_MAP
from src.services.name_to_code_resolver import resolve_name_to_code


_SEPARATORS = re.compile(r"[\s,，;；、]+")
_CRYPTO_ALIASES = {
    "btc": "BTC",
    "bitcoin": "BTC",
    "比特币": "BTC",
    "eth": "ETH",
    "ethereum": "ETH",
    "以太坊": "ETH",
}


@dataclass
class GitHubActionsClient:
    """Small REST client for repository actions variables and workflow dispatch."""

    repo: str
    token: str
    api_url: str = "https://api.github.com"

    @classmethod
    def from_env(cls) -> "GitHubActionsClient":
        repo = (
            os.getenv("WATCHLIST_GITHUB_REPOSITORY")
            or os.getenv("GITHUB_REPOSITORY")
            or "Chuuuyy/daily_stock_monitor"
        ).strip()
        token = (
            os.getenv("WATCHLIST_GITHUB_TOKEN")
            or os.getenv("GITHUB_PAT")
            or os.getenv("GITHUB_TOKEN")
            or ""
        ).strip()
        if not token:
            raise RuntimeError("缺少 GitHub Token。请配置 WATCHLIST_GITHUB_TOKEN 或 GITHUB_PAT。")
        return cls(repo=repo, token=token)

    @property
    def _headers(self) -> dict:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def get_variable(self, name: str) -> Optional[str]:
        url = f"{self.api_url}/repos/{self.repo}/actions/variables/{name}"
        response = requests.get(url, headers=self._headers, timeout=15)
        if response.status_code == 404:
            return None
        self._raise_for_status(response)
        return str(response.json().get("value") or "")

    def set_variable(self, name: str, value: str) -> None:
        url = f"{self.api_url}/repos/{self.repo}/actions/variables/{name}"
        payload = {"name": name, "value": value}
        response = requests.patch(url, headers=self._headers, json=payload, timeout=15)
        if response.status_code == 404:
            response = requests.post(
                f"{self.api_url}/repos/{self.repo}/actions/variables",
                headers=self._headers,
                json=payload,
                timeout=15,
            )
        self._raise_for_status(response)

    def dispatch_daily_monitor(self) -> str:
        workflow = os.getenv("WATCHLIST_DAILY_WORKFLOW", "00-daily-analysis.yml").strip()
        ref = (
            os.getenv("WATCHLIST_WORKFLOW_REF")
            or os.getenv("GITHUB_REF_NAME")
            or "main"
        ).strip()
        url = f"{self.api_url}/repos/{self.repo}/actions/workflows/{workflow}/dispatches"
        payload = {
            "ref": ref,
            "inputs": {
                "mode": "stocks-only",
                "force_run": "true",
            },
        }
        response = requests.post(url, headers=self._headers, json=payload, timeout=15)
        self._raise_for_status(response)
        return ref

    def _raise_for_status(self, response: requests.Response) -> None:
        if 200 <= response.status_code < 300:
            return
        try:
            message = response.json().get("message") or response.text
        except Exception:
            message = response.text
        raise RuntimeError(f"GitHub API {response.status_code}: {message[:200]}")


class WatchlistCommand(BotCommand):
    """Admin command for changing the scheduled daily monitor watchlist."""

    @property
    def name(self) -> str:
        return "watchlist"

    @property
    def aliases(self) -> List[str]:
        return ["watch", "自选", "自选股", "关注"]

    @property
    def description(self) -> str:
        return "管理每日分析自选股"

    @property
    def usage(self) -> str:
        return "/watchlist [list|set|add|remove|run] <股票代码或名称...>"

    @property
    def admin_only(self) -> bool:
        return True

    def execute(self, message: BotMessage, args: List[str]) -> BotResponse:
        action, raw_items = self._parse_action(args)

        if action == "list":
            return self._show()
        if action == "run":
            return self._run()

        current = self._get_current_codes()
        if action == "remove":
            codes = self._resolve_items(raw_items)
            updated = [code for code in current if code not in set(codes)]
            return self._save(updated, "已移除")

        if action == "add":
            codes = self._resolve_items(raw_items)
            updated = list(current)
            for code in codes:
                if code not in updated:
                    updated.append(code)
            return self._save(updated, "已添加")

        codes = self._resolve_items(raw_items)
        return self._save(codes, "已更新")

    def validate_args(self, args: List[str]) -> Optional[str]:
        action, raw_items = self._parse_action(args)
        if action in {"list", "run"}:
            return None
        if not raw_items:
            return "请提供至少一个股票代码或名称"
        return None

    def _parse_action(self, args: List[str]) -> tuple[str, List[str]]:
        if not args:
            return "list", []
        first = args[0].strip().lower()
        aliases = {
            "list": "list",
            "show": "list",
            "查看": "list",
            "set": "set",
            "设置": "set",
            "更新": "set",
            "add": "add",
            "添加": "add",
            "加": "add",
            "remove": "remove",
            "rm": "remove",
            "del": "remove",
            "删除": "remove",
            "移除": "remove",
            "run": "run",
            "分析": "run",
            "跑": "run",
        }
        if first in aliases:
            return aliases[first], args[1:]
        return "set", args

    def _split_items(self, items: Iterable[str]) -> List[str]:
        result: List[str] = []
        for item in items:
            for part in _SEPARATORS.split(item.strip()):
                if part:
                    result.append(part)
        return result

    def _resolve_items(self, items: Iterable[str]) -> List[str]:
        resolved: List[str] = []
        failed: List[str] = []

        for item in self._split_items(items):
            code = _CRYPTO_ALIASES.get(item.lower()) or resolve_name_to_code(item)
            if not code:
                failed.append(item)
                continue
            code = code.upper()
            if code not in resolved:
                resolved.append(code)

        if failed:
            raise ValueError("无法识别: " + "、".join(failed))
        if not resolved:
            raise ValueError("没有可用的股票代码")
        return resolved

    def _get_client(self) -> GitHubActionsClient:
        return GitHubActionsClient.from_env()

    def _get_current_codes(self) -> List[str]:
        try:
            value = self._get_client().get_variable("STOCK_LIST")
        except Exception:
            value = os.getenv("STOCK_LIST", "")
        return [item.strip().upper() for item in (value or "").split(",") if item.strip()]

    def _show(self) -> BotResponse:
        codes = self._get_current_codes()
        if not codes:
            return BotResponse.markdown_response("当前自选股为空。")
        return BotResponse.markdown_response(self._format_codes("当前自选股", codes))

    def _run(self) -> BotResponse:
        ref = self._get_client().dispatch_daily_monitor()
        return BotResponse.markdown_response(
            "✅ 已触发 daily monitor\n\n"
            f"• 分支: `{ref}`\n"
            "• 模式: `stocks-only`"
        )

    def _save(self, codes: List[str], title: str) -> BotResponse:
        value = ",".join(codes)
        self._get_client().set_variable("STOCK_LIST", value)
        os.environ["STOCK_LIST"] = value
        try:
            from src.config import get_config

            get_config().refresh_stock_list()
        except Exception:
            pass
        return BotResponse.markdown_response(self._format_codes(title, codes))

    def _format_codes(self, title: str, codes: List[str]) -> str:
        lines = [f"✅ **{title}**", ""]
        for code in codes:
            name = STOCK_NAME_MAP.get(code, "")
            suffix = f" {name}" if name else ""
            lines.append(f"• `{code}`{suffix}")
        lines.extend(["", f"GitHub Actions 变量 `STOCK_LIST` = `{','.join(codes)}`"])
        return "\n".join(lines)
