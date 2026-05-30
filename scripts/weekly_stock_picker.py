#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Weekly A-share candidate screener.

This script is intentionally token-light: it ranks the full A-share market with
market-data rules first, then optionally spends a small number of Tavily calls to
enrich only the final candidates with recent news.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.notification import get_notification_service  # noqa: E402


LOGGER = logging.getLogger("weekly_stock_picker")
MARKET_TZ = ZoneInfo(os.getenv("WEEKLY_PICKER_TIMEZONE", "Asia/Shanghai"))


def market_now() -> datetime:
    return datetime.now(MARKET_TZ)


@dataclass
class PickerConfig:
    top_n: int = 3
    min_amount_yi: float = 1.5
    min_turnover: float = 1.0
    max_turnover: float = 20.0
    min_price: float = 3.0
    max_price: float = 200.0
    sector_count: int = 8
    max_per_sector: int = 1
    news_enabled: bool = True
    news_days: int = 14
    news_results: int = 3
    send_notification: bool = True


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    raw = os.getenv(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        LOGGER.warning("%s=%r is invalid; using %s", name, raw, default)
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _env_float(name: str, default: float, *, minimum: Optional[float] = None) -> float:
    raw = os.getenv(name, "").strip()
    try:
        value = float(raw) if raw else default
    except ValueError:
        LOGGER.warning("%s=%r is invalid; using %s", name, raw, default)
        value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def load_config(args: argparse.Namespace) -> PickerConfig:
    return PickerConfig(
        top_n=args.top_n or _env_int("WEEKLY_PICKER_TOP_N", 3, minimum=1, maximum=30),
        min_amount_yi=_env_float("WEEKLY_PICKER_MIN_AMOUNT_YI", 1.5, minimum=0.0),
        min_turnover=_env_float("WEEKLY_PICKER_MIN_TURNOVER", 1.0, minimum=0.0),
        max_turnover=_env_float("WEEKLY_PICKER_MAX_TURNOVER", 20.0, minimum=0.0),
        min_price=_env_float("WEEKLY_PICKER_MIN_PRICE", 3.0, minimum=0.0),
        max_price=_env_float("WEEKLY_PICKER_MAX_PRICE", 200.0, minimum=0.0),
        sector_count=_env_int("WEEKLY_PICKER_SECTOR_COUNT", 8, minimum=3, maximum=20),
        max_per_sector=_env_int("WEEKLY_PICKER_MAX_PER_SECTOR", 1, minimum=1, maximum=5),
        news_enabled=args.news if args.news is not None else _env_bool("WEEKLY_PICKER_NEWS_ENABLED", True),
        news_days=_env_int("WEEKLY_PICKER_NEWS_DAYS", 14, minimum=1, maximum=60),
        news_results=_env_int("WEEKLY_PICKER_NEWS_RESULTS", 3, minimum=1, maximum=5),
        send_notification=args.send if args.send is not None else _env_bool("WEEKLY_PICKER_SEND", True),
    )


def _to_number(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "").replace("%", "")
    if text in {"", "-", "--", "None", "nan"}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _first_present(columns: Iterable[str], *candidates: str) -> Optional[str]:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def fetch_market_snapshot() -> pd.DataFrame:
    errors: List[str] = []
    tushare_token = os.getenv("TUSHARE_TOKEN", "").strip()

    if tushare_token:
        try:
            df = _fetch_tushare_snapshot(tushare_token)
            if df is not None and not df.empty:
                return df
            errors.append("tushare returned empty snapshot")
        except Exception as exc:
            errors.append(f"tushare fallback: {exc}")
            LOGGER.warning("Tushare snapshot fetch failed: %s", exc)

    try:
        df = _fetch_tencent_snapshot()
        if df is not None and not df.empty:
            return df
        errors.append("tencent returned empty snapshot")
    except Exception as exc:
        errors.append(f"tencent fallback: {exc}")
        LOGGER.warning("Tencent snapshot fallback failed: %s", exc)

    try:
        import akshare as ak

        for attempt in range(1, 4):
            try:
                LOGGER.info("Fetching A-share snapshot via akshare.stock_zh_a_spot_em, attempt %s/3", attempt)
                df = ak.stock_zh_a_spot_em()
                if df is not None and not df.empty:
                    return df
                errors.append("akshare returned empty snapshot")
            except Exception as exc:
                errors.append(f"akshare attempt {attempt}: {exc}")
                LOGGER.warning("AkShare snapshot attempt %s failed: %s", attempt, exc)
                time.sleep(attempt * 2)
    except ImportError as exc:
        errors.append(f"akshare import failed: {exc}")

    try:
        import efinance as ef

        LOGGER.info("Fetching A-share snapshot via efinance.stock.get_realtime_quotes fallback")
        df = ef.stock.get_realtime_quotes()
        if df is not None and not df.empty:
            return df
        errors.append("efinance returned empty snapshot")
    except Exception as exc:
        errors.append(f"efinance fallback: {exc}")
        LOGGER.warning("Efinance snapshot fallback failed: %s", exc)

    raise RuntimeError("Unable to fetch A-share snapshot: " + " | ".join(errors[-5:]))


def _to_tencent_symbol(code: str) -> str:
    return f"sh{code}" if code.startswith("6") else f"sz{code}"


def _parse_tencent_amount(fields: List[str]) -> float:
    if len(fields) > 35 and fields[35]:
        parts = fields[35].split("/")
        if len(parts) >= 3:
            amount = _to_number(parts[2])
            if pd.notna(amount):
                return amount
    if len(fields) > 37:
        amount_wan = _to_number(fields[37])
        if pd.notna(amount_wan):
            return amount_wan * 10000
    return float("nan")


def _fetch_tencent_snapshot() -> pd.DataFrame:
    index_path = ROOT / "apps" / "dsa-web" / "public" / "stocks.index.json"
    if not index_path.exists():
        raise RuntimeError(f"stock index not found: {index_path}")

    entries = json.loads(index_path.read_text(encoding="utf-8"))
    stocks = [
        {"code": str(item[1]), "name": str(item[2])}
        for item in entries
        if len(item) >= 8 and item[6] == "CN" and item[7] == "stock" and str(item[1]).isdigit()
    ]
    if not stocks:
        raise RuntimeError("stock index has no CN stock entries")

    LOGGER.info("Fetching A-share snapshot via Tencent batch quotes, stocks=%s", len(stocks))
    rows: List[Dict[str, Any]] = []
    headers = {
        "Referer": "https://finance.qq.com/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    }
    for start in range(0, len(stocks), 120):
        batch = stocks[start : start + 120]
        query = ",".join(_to_tencent_symbol(item["code"]) for item in batch)
        response = requests.get(f"http://qt.gtimg.cn/q={query}", headers=headers, timeout=15)
        response.encoding = "gbk"
        if response.status_code != 200:
            LOGGER.warning("Tencent batch %s failed: HTTP %s", start // 120 + 1, response.status_code)
            continue

        for line in response.text.splitlines():
            if '="' not in line:
                continue
            data_start = line.find('"')
            data_end = line.rfind('"')
            if data_start < 0 or data_end <= data_start:
                continue
            fields = line[data_start + 1 : data_end].split("~")
            if len(fields) < 39 or not fields[2]:
                continue
            code = str(fields[2]).strip()
            price = _to_number(fields[3] if len(fields) > 3 else None)
            pct = _to_number(fields[32] if len(fields) > 32 else None)
            amount = _parse_tencent_amount(fields)
            if not code or pd.isna(price) or pd.isna(pct) or pd.isna(amount):
                continue
            rows.append(
                {
                    "代码": code,
                    "名称": fields[1] if len(fields) > 1 and fields[1] else code,
                    "最新价": price,
                    "涨跌幅": pct,
                    "成交额": amount,
                    "换手率": _to_number(fields[38] if len(fields) > 38 else None),
                    "量比": _to_number(fields[49] if len(fields) > 49 else None),
                    "市盈率-动态": _to_number(fields[39] if len(fields) > 39 else None),
                    "市净率": _to_number(fields[46] if len(fields) > 46 else None),
                }
            )
        time.sleep(0.15)

    if not rows:
        raise RuntimeError("Tencent quote batches returned no parseable rows")
    LOGGER.info("Tencent snapshot returned %s rows", len(rows))
    return pd.DataFrame(rows)


def _fetch_tushare_snapshot(token: str) -> pd.DataFrame:
    import tushare as ts

    LOGGER.info("Fetching A-share snapshot via Tushare daily_basic fallback")
    pro = ts.pro_api(token)
    today = market_now()
    daily = pd.DataFrame()
    trade_date = ""
    errors: List[str] = []
    for offset in range(0, 15):
        candidate_date = (today - timedelta(days=offset)).strftime("%Y%m%d")
        try:
            candidate_daily = pro.daily(trade_date=candidate_date)
        except Exception as exc:
            errors.append(f"{candidate_date}: {exc}")
            continue
        if candidate_daily is not None and not candidate_daily.empty:
            daily = candidate_daily
            trade_date = candidate_date
            break
    if daily.empty:
        raise RuntimeError("Tushare daily is empty for recent dates: " + " | ".join(errors[-5:]))
    LOGGER.info("Using Tushare trade_date=%s", trade_date)

    try:
        basic = pro.daily_basic(
            trade_date=trade_date,
            fields="ts_code,turnover_rate,volume_ratio,pe,pb,total_mv,circ_mv",
        )
    except Exception as exc:
        LOGGER.warning("Tushare daily_basic unavailable, continuing with daily data only: %s", exc)
        basic = pd.DataFrame({"ts_code": daily["ts_code"]})

    try:
        stocks = pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name,market",
        )
    except Exception as exc:
        LOGGER.warning("Tushare stock_basic unavailable, using stock code as name: %s", exc)
        stocks = pd.DataFrame({"ts_code": daily["ts_code"], "name": daily["ts_code"]})

    merged = daily.merge(basic, on="ts_code", how="left").merge(stocks, on="ts_code", how="left")
    result = pd.DataFrame(
        {
            "代码": merged["ts_code"].astype(str).str.extract(r"(\d{6})", expand=False),
            "名称": merged.get("name", merged["ts_code"]),
            "最新价": merged["close"],
            "涨跌幅": merged["pct_chg"],
            "成交额": merged["amount"].map(_to_number) * 1000,
            "换手率": merged["turnover_rate"] if "turnover_rate" in merged else float("nan"),
            "量比": merged["volume_ratio"] if "volume_ratio" in merged else float("nan"),
            "市盈率-动态": merged["pe"] if "pe" in merged else float("nan"),
            "市净率": merged["pb"] if "pb" in merged else float("nan"),
        }
    )
    return result.dropna(subset=["代码", "名称", "最新价", "涨跌幅", "成交额"])


def normalize_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    column_map = {
        "code": _first_present(df.columns, "代码", "code", "股票代码"),
        "name": _first_present(df.columns, "名称", "name", "股票名称"),
        "price": _first_present(df.columns, "最新价", "最新", "close"),
        "pct": _first_present(df.columns, "涨跌幅", "changepercent", "涨幅"),
        "amount": _first_present(df.columns, "成交额", "amount"),
        "turnover": _first_present(df.columns, "换手率", "turnoverratio", "换手"),
        "volume_ratio": _first_present(df.columns, "量比", "volume_ratio"),
        "pe": _first_present(df.columns, "市盈率-动态", "市盈率", "pe", "PE"),
        "pb": _first_present(df.columns, "市净率", "pb", "PB"),
    }
    missing = [key for key, value in column_map.items() if value is None and key in {"code", "name", "price", "pct", "amount"}]
    if missing:
        raise RuntimeError(f"Snapshot missing required columns: {missing}; got {list(df.columns)}")

    normalized = pd.DataFrame()
    for field, source in column_map.items():
        if source is None:
            normalized[field] = float("nan")
        else:
            normalized[field] = df[source]

    normalized["code"] = normalized["code"].astype(str).str.extract(r"(\d{6})", expand=False)
    normalized["name"] = normalized["name"].astype(str).str.strip()
    for col in ["price", "pct", "amount", "turnover", "volume_ratio", "pe", "pb"]:
        normalized[col] = normalized[col].map(_to_number)

    normalized = normalized.dropna(subset=["code", "name", "price", "pct", "amount"])
    normalized["amount_yi"] = normalized["amount"] / 100000000
    return normalized


def fetch_sector_members(config: PickerConfig) -> Dict[str, str]:
    """Return code -> sector for the strongest recent industry boards."""
    try:
        import akshare as ak
    except ImportError:
        LOGGER.warning("akshare is not installed; skip sector membership")
        return {}

    try:
        sector_df = ak.stock_board_industry_name_em()
    except Exception as exc:
        LOGGER.warning("Fetch industry sector ranking failed: %s", exc)
        return {}
    if sector_df is None or sector_df.empty:
        return {}

    name_col = _first_present(sector_df.columns, "板块名称", "板块", "行业名称", "名称")
    pct_col = _first_present(sector_df.columns, "涨跌幅", "涨幅", "change_pct")
    if not name_col:
        return {}

    work = sector_df.copy()
    if pct_col:
        work["_pct"] = work[pct_col].map(_to_number)
        work = work.sort_values("_pct", ascending=False)

    mapping: Dict[str, str] = {}
    sector_names = [str(name).strip() for name in work[name_col].head(config.sector_count) if str(name).strip()]
    for sector_name in sector_names:
        try:
            cons = ak.stock_board_industry_cons_em(symbol=sector_name)
        except Exception as exc:
            LOGGER.debug("Fetch sector constituents failed for %s: %s", sector_name, exc)
            continue
        if cons is None or cons.empty:
            continue
        code_col = _first_present(cons.columns, "代码", "股票代码", "code")
        if not code_col:
            continue
        for raw_code in cons[code_col]:
            code_match = re.search(r"\d{6}", str(raw_code))
            if code_match:
                mapping.setdefault(code_match.group(0), sector_name)
        time.sleep(0.2)

    LOGGER.info("Loaded sector membership: sectors=%s stocks=%s", len(sector_names), len(mapping))
    return mapping


def _score_row(row: pd.Series) -> tuple[float, List[str], List[str]]:
    score = 50.0
    reasons: List[str] = []
    risks: List[str] = []

    pct = row["pct"]
    if 0.5 <= pct <= 4.5:
        score += 10 + min(pct, 4.5)
        reasons.append("温和放量上涨")
    elif -1.5 <= pct < 0.5:
        score += 4
        reasons.append("波动较稳")
    elif pct > 6.5:
        score -= 12
        risks.append("短线涨幅偏高")
    else:
        score -= 8
        risks.append("当日价格表现偏弱")

    turnover = row["turnover"]
    if 2 <= turnover <= 8:
        score += 12
        reasons.append("换手活跃但未过热")
    elif 1 <= turnover < 2 or 8 < turnover <= 12:
        score += 6
    elif turnover > 15:
        score -= 8
        risks.append("换手过热")

    amount_yi = row["amount_yi"]
    if amount_yi >= 3:
        score += min(12, math.log10(amount_yi) * 8)
        reasons.append("成交额支撑较好")
    elif amount_yi < 1.5:
        score -= 8
        risks.append("成交额不足")

    volume_ratio = row["volume_ratio"]
    if pd.notna(volume_ratio):
        if 1.0 <= volume_ratio <= 2.8:
            score += 8
            reasons.append("量比健康")
        elif volume_ratio > 4:
            score -= 6
            risks.append("量比异常偏高")

    pe = row["pe"]
    if pd.notna(pe):
        if 0 < pe <= 45:
            score += 6
            reasons.append("估值未明显失控")
        elif pe > 90 or pe <= 0:
            score -= 8
            risks.append("PE 异常或过高")

    pb = row["pb"]
    if pd.notna(pb):
        if 0 < pb <= 8:
            score += 4
        elif pb > 12:
            score -= 5
            risks.append("PB 偏高")

    sector = str(row.get("sector") or "").strip()
    if sector and sector != "未分类":
        score += 6
        reasons.append(f"处于强势板块：{sector}")

    return round(max(0, min(100, score)), 1), reasons[:4], risks[:4]


def _diversify_by_sector(data: pd.DataFrame, config: PickerConfig) -> pd.DataFrame:
    selected: List[int] = []
    sector_counts: Dict[str, int] = {}

    for idx, row in data.iterrows():
        sector = str(row.get("sector") or "未分类")
        count = sector_counts.get(sector, 0)
        if sector != "未分类" and count >= config.max_per_sector:
            continue
        selected.append(idx)
        sector_counts[sector] = count + 1
        if len(selected) >= config.top_n:
            break

    if len(selected) < config.top_n:
        for idx in data.index:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= config.top_n:
                break

    return data.loc[selected].reset_index(drop=True)


def build_candidates(df: pd.DataFrame, config: PickerConfig, sector_members: Dict[str, str]) -> pd.DataFrame:
    data = df.copy()
    data = data[data["code"].str.match(r"^[036]\d{5}$", na=False)]
    data = data[~data["name"].str.contains("ST|退|退市", case=False, na=False)]
    data = data[(data["price"] >= config.min_price) & (data["price"] <= config.max_price)]
    data = data[data["amount_yi"] >= config.min_amount_yi]
    data = data[(data["turnover"].isna()) | ((data["turnover"] >= config.min_turnover) & (data["turnover"] <= config.max_turnover))]
    data = data[(data["pct"] >= -4.5) & (data["pct"] <= 7.5)]
    data["sector"] = data["code"].map(sector_members).fillna("未分类")

    if sector_members:
        sector_data = data[data["sector"] != "未分类"].copy()
        if not sector_data.empty:
            data = sector_data

    scored = data.apply(_score_row, axis=1, result_type="expand")
    data["score"] = scored[0]
    data["reasons"] = scored[1]
    data["risks"] = scored[2]
    ranked = data.sort_values(["score", "amount_yi"], ascending=[False, False])
    return _diversify_by_sector(ranked, config)


def search_recent_news(candidates: pd.DataFrame, config: PickerConfig) -> Dict[str, List[Dict[str, str]]]:
    keys = [item.strip() for item in os.getenv("TAVILY_API_KEYS", "").split(",") if item.strip()]
    if not config.news_enabled or not keys:
        return {}

    try:
        from tavily import TavilyClient
    except ImportError:
        LOGGER.warning("tavily-python is not installed; skipping news enrichment")
        return {}

    client = TavilyClient(api_key=keys[0])
    news_by_code: Dict[str, List[Dict[str, str]]] = {}
    for _, row in candidates.iterrows():
        query = f'{row["name"]} {row["code"]} 财报 业绩 订单 产业 新闻'
        try:
            response = client.search(
                query=query,
                search_depth="basic",
                max_results=config.news_results,
                include_answer=False,
                include_raw_content=False,
                days=config.news_days,
                topic="news",
            )
        except Exception as exc:  # pragma: no cover - depends on external API
            LOGGER.warning("Tavily search failed for %s(%s): %s", row["name"], row["code"], exc)
            continue
        results: List[Dict[str, str]] = []
        for item in response.get("results", []):
            results.append(
                {
                    "title": str(item.get("title") or "").strip(),
                    "url": str(item.get("url") or "").strip(),
                    "content": str(item.get("content") or "").strip()[:120],
                }
            )
        news_by_code[str(row["code"])] = results
    return news_by_code


def _fmt(value: Any, suffix: str = "", default: str = "-") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def render_report(candidates: pd.DataFrame, news_by_code: Dict[str, List[Dict[str, str]]]) -> str:
    now = market_now()
    lines = [
        "# AI 每周股票候选观察池",
        "",
        f"生成时间：{now.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "> 说明：这是基于公开行情、流动性、量价和估值约束生成的研究候选池，不构成投资建议。买入前仍需要结合仓位、止损、财报质量和市场环境独立判断。",
        "",
        "## 候选列表",
        "",
        "| 排名 | 板块 | 代码 | 名称 | 评分 | 最新价 | 涨跌幅 | 成交额(亿) | 换手率 | PE | PB | 主要理由 | 风险点 |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for idx, row in candidates.iterrows():
        reasons = "；".join(row["reasons"]) if row["reasons"] else "-"
        risks = "；".join(row["risks"]) if row["risks"] else "暂无明显规则风险"
        lines.append(
            "| {rank} | {sector} | {code} | {name} | {score} | {price} | {pct} | {amount} | {turnover} | {pe} | {pb} | {reasons} | {risks} |".format(
                rank=idx + 1,
                sector=row.get("sector") or "未分类",
                code=row["code"],
                name=row["name"],
                score=_fmt(row["score"]),
                price=_fmt(row["price"]),
                pct=_fmt(row["pct"], "%"),
                amount=_fmt(row["amount_yi"]),
                turnover=_fmt(row["turnover"], "%"),
                pe=_fmt(row["pe"]),
                pb=_fmt(row["pb"]),
                reasons=reasons,
                risks=risks,
            )
        )

    lines.extend(["", "## 逐只观察要点", ""])
    for idx, row in candidates.iterrows():
        lines.extend(
            [
                f"### {idx + 1}. {row['name']}（{row['code']}）",
                "",
                f"- 所属主线：{row.get('sector') or '未分类'}",
                f"- 规则评分：{_fmt(row['score'])}",
                f"- 量价状态：涨跌幅 {_fmt(row['pct'], '%')}，成交额 {_fmt(row['amount_yi'])} 亿，换手率 {_fmt(row['turnover'], '%')}，量比 {_fmt(row['volume_ratio'])}",
                f"- 估值观察：PE {_fmt(row['pe'])}，PB {_fmt(row['pb'])}",
                f"- 入池理由：{'；'.join(row['reasons']) if row['reasons'] else '基础筛选通过'}",
                f"- 需要警惕：{'；'.join(row['risks']) if row['risks'] else '未触发主要规则风险'}",
            ]
        )
        news_items = news_by_code.get(str(row["code"]), [])
        if news_items:
            lines.append("- 近期信息：")
            for item in news_items:
                title = item["title"] or item["content"] or "相关资讯"
                url = item["url"]
                lines.append(f"  - [{title}]({url})" if url else f"  - {title}")
        else:
            lines.append("- 近期信息：未补充或未检索到高相关结果")
        lines.append("")

    lines.extend(
        [
            "## 使用方式",
            "",
            "- 更偏保守：提高 `WEEKLY_PICKER_MIN_AMOUNT_YI`，降低 `WEEKLY_PICKER_MAX_TURNOVER`。",
            "- 更少消耗：把 `WEEKLY_PICKER_NEWS_ENABLED` 设为 `false`，只保留行情规则筛选。",
            "- 更集中：默认只推送 3 只；如需扩大观察池，再调高 `WEEKLY_PICKER_TOP_N`。",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _snapshot_payload(candidates: pd.DataFrame) -> Dict[str, Any]:
    generated_at = market_now().isoformat()
    rows: List[Dict[str, Any]] = []
    for idx, row in candidates.iterrows():
        rows.append(
            {
                "rank": int(idx + 1),
                "code": str(row["code"]),
                "name": str(row["name"]),
                "sector": str(row.get("sector") or "未分类"),
                "price": float(row["price"]),
                "pct": float(row["pct"]),
                "score": float(row["score"]),
            }
        )
    return {"generated_at": generated_at, "candidates": rows}


def _latest_previous_snapshot() -> Optional[Dict[str, Any]]:
    reports_dir = ROOT / "reports"
    if not reports_dir.exists():
        return None
    today = market_now().strftime("%Y%m%d")
    paths = sorted(
        p for p in reports_dir.glob("weekly_stock_picker_*.json")
        if today not in p.stem
    )
    if not paths:
        return None
    try:
        return json.loads(paths[-1].read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Failed to load previous weekly snapshot %s: %s", paths[-1], exc)
        return None


def build_review_section(previous: Optional[Dict[str, Any]], snapshot: pd.DataFrame) -> str:
    if not previous or not previous.get("candidates"):
        return "## 上期候选复盘\n\n- 暂无上期结构化快照，本期开始记录，下周会自动复盘。\n"

    current = snapshot.set_index("code")
    lines = [
        "## 上期候选复盘",
        "",
        f"上期生成时间：{previous.get('generated_at', '-')}",
        "",
        "| 上期排名 | 板块 | 代码 | 名称 | 上期价格 | 当前价格 | 区间变化 | 今日涨跌 | 复盘结论 |",
        "|---:|---|---|---|---:|---:|---:|---:|---|",
    ]
    hits = 0
    total = 0
    for item in previous.get("candidates", []):
        code = str(item.get("code") or "")
        if code not in current.index:
            continue
        total += 1
        row = current.loc[code]
        old_price = _to_number(item.get("price"))
        new_price = _to_number(row.get("price"))
        change = float("nan")
        if pd.notna(old_price) and old_price > 0 and pd.notna(new_price):
            change = (new_price / old_price - 1) * 100
        today_pct = _to_number(row.get("pct"))
        ok = pd.notna(change) and change >= 0
        if ok:
            hits += 1
        conclusion = "符合/偏强" if ok else "未兑现/需降权"
        lines.append(
            "| {rank} | {sector} | {code} | {name} | {old} | {new} | {change} | {today} | {conclusion} |".format(
                rank=item.get("rank", "-"),
                sector=item.get("sector") or "未分类",
                code=code,
                name=item.get("name") or row.get("name") or "-",
                old=_fmt(old_price),
                new=_fmt(new_price),
                change=_fmt(change, "%"),
                today=_fmt(today_pct, "%"),
                conclusion=conclusion,
            )
        )

    if total:
        lines.extend(["", f"- 简评：上期 {total} 只候选中，价格正向兑现 {hits} 只。"])
    else:
        lines.extend(["", "- 简评：上期候选未能在当前行情快照中匹配到，暂不评价。"])
    return "\n".join(lines) + "\n"


def save_report(content: str, candidates: pd.DataFrame) -> Path:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = market_now().strftime("%Y%m%d")
    path = reports_dir / f"weekly_stock_picker_{stamp}.md"
    path.write_text(content, encoding="utf-8")
    (reports_dir / f"weekly_stock_picker_{stamp}.json").write_text(
        json.dumps(_snapshot_payload(candidates), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a weekly stock candidate watchlist")
    parser.add_argument("--top-n", type=int, default=None, help="number of candidates to keep")
    parser.add_argument("--news", dest="news", action="store_true", default=None, help="enable Tavily news enrichment")
    parser.add_argument("--no-news", dest="news", action="store_false", help="disable Tavily news enrichment")
    parser.add_argument("--send", dest="send", action="store_true", default=None, help="send notification")
    parser.add_argument("--no-send", dest="send", action="store_false", help="do not send notification")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    config = load_config(args)
    LOGGER.info("Weekly picker config: %s", config)

    snapshot = normalize_snapshot(fetch_market_snapshot())
    previous = _latest_previous_snapshot()
    sector_members = fetch_sector_members(config)
    candidates = build_candidates(snapshot, config, sector_members)
    if candidates.empty:
        raise RuntimeError("No candidates passed the weekly picker filters")

    news_by_code = search_recent_news(candidates, config)
    report = build_review_section(previous, snapshot) + "\n" + render_report(candidates, news_by_code)
    report_path = save_report(report, candidates)
    LOGGER.info("Weekly stock picker report saved: %s", report_path)

    if config.send_notification:
        service = get_notification_service()
        if service.is_available():
            ok = service.send(report, route_type="report", severity="info")
            LOGGER.info("Notification sent: %s", ok)
        else:
            LOGGER.warning("No notification channel configured; skip sending")

    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
