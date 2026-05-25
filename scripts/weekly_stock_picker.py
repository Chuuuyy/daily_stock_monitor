#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Weekly A-share candidate screener.

This script is intentionally token-light: it ranks the full A-share market with
market-data rules first, then optionally spends a small number of Tavily calls to
enrich only the final candidates with recent news.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.notification import get_notification_service  # noqa: E402


LOGGER = logging.getLogger("weekly_stock_picker")


@dataclass
class PickerConfig:
    top_n: int = 10
    min_amount_yi: float = 1.5
    min_turnover: float = 1.0
    max_turnover: float = 20.0
    min_price: float = 3.0
    max_price: float = 200.0
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
        top_n=args.top_n or _env_int("WEEKLY_PICKER_TOP_N", 10, minimum=3, maximum=30),
        min_amount_yi=_env_float("WEEKLY_PICKER_MIN_AMOUNT_YI", 1.5, minimum=0.0),
        min_turnover=_env_float("WEEKLY_PICKER_MIN_TURNOVER", 1.0, minimum=0.0),
        max_turnover=_env_float("WEEKLY_PICKER_MAX_TURNOVER", 20.0, minimum=0.0),
        min_price=_env_float("WEEKLY_PICKER_MIN_PRICE", 3.0, minimum=0.0),
        max_price=_env_float("WEEKLY_PICKER_MAX_PRICE", 200.0, minimum=0.0),
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


def _fetch_tushare_snapshot(token: str) -> pd.DataFrame:
    import tushare as ts

    LOGGER.info("Fetching A-share snapshot via Tushare daily_basic fallback")
    pro = ts.pro_api(token)
    today = datetime.now().strftime("%Y%m%d")
    calendar = pro.trade_cal(exchange="SSE", start_date="20200101", end_date=today, is_open="1")
    if calendar is None or calendar.empty:
        raise RuntimeError("Tushare trade calendar is empty")
    trade_date = str(calendar.sort_values("cal_date").iloc[-1]["cal_date"])
    LOGGER.info("Using Tushare trade_date=%s", trade_date)

    daily = pro.daily(trade_date=trade_date)
    basic = pro.daily_basic(
        trade_date=trade_date,
        fields="ts_code,turnover_rate,volume_ratio,pe,pb,total_mv,circ_mv",
    )
    stocks = pro.stock_basic(
        exchange="",
        list_status="L",
        fields="ts_code,symbol,name,market",
    )
    if daily is None or daily.empty:
        raise RuntimeError(f"Tushare daily is empty for {trade_date}")
    merged = daily.merge(basic, on="ts_code", how="left").merge(stocks, on="ts_code", how="left")
    result = pd.DataFrame(
        {
            "代码": merged["ts_code"].astype(str).str.extract(r"(\d{6})", expand=False),
            "名称": merged["name"],
            "最新价": merged["close"],
            "涨跌幅": merged["pct_chg"],
            "成交额": merged["amount"].map(_to_number) * 1000,
            "换手率": merged["turnover_rate"],
            "量比": merged["volume_ratio"],
            "市盈率-动态": merged["pe"],
            "市净率": merged["pb"],
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

    return round(max(0, min(100, score)), 1), reasons[:4], risks[:4]


def build_candidates(df: pd.DataFrame, config: PickerConfig) -> pd.DataFrame:
    data = df.copy()
    data = data[data["code"].str.match(r"^[036]\d{5}$", na=False)]
    data = data[~data["name"].str.contains("ST|退|退市", case=False, na=False)]
    data = data[(data["price"] >= config.min_price) & (data["price"] <= config.max_price)]
    data = data[data["amount_yi"] >= config.min_amount_yi]
    data = data[(data["turnover"].isna()) | ((data["turnover"] >= config.min_turnover) & (data["turnover"] <= config.max_turnover))]
    data = data[(data["pct"] >= -4.5) & (data["pct"] <= 7.5)]

    scored = data.apply(_score_row, axis=1, result_type="expand")
    data["score"] = scored[0]
    data["reasons"] = scored[1]
    data["risks"] = scored[2]
    return data.sort_values(["score", "amount_yi"], ascending=[False, False]).head(config.top_n).reset_index(drop=True)


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
    now = datetime.now()
    lines = [
        "# AI 每周股票候选观察池",
        "",
        f"生成时间：{now.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "> 说明：这是基于公开行情、流动性、量价和估值约束生成的研究候选池，不构成投资建议。买入前仍需要结合仓位、止损、财报质量和市场环境独立判断。",
        "",
        "## 候选列表",
        "",
        "| 排名 | 代码 | 名称 | 评分 | 涨跌幅 | 成交额(亿) | 换手率 | PE | PB | 主要理由 | 风险点 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for idx, row in candidates.iterrows():
        reasons = "；".join(row["reasons"]) if row["reasons"] else "-"
        risks = "；".join(row["risks"]) if row["risks"] else "暂无明显规则风险"
        lines.append(
            "| {rank} | {code} | {name} | {score} | {pct} | {amount} | {turnover} | {pe} | {pb} | {reasons} | {risks} |".format(
                rank=idx + 1,
                code=row["code"],
                name=row["name"],
                score=_fmt(row["score"]),
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
            "- 更集中：把 `WEEKLY_PICKER_TOP_N` 设为 5，只推送最靠前的候选。",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def save_report(content: str) -> Path:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"weekly_stock_picker_{datetime.now().strftime('%Y%m%d')}.md"
    path.write_text(content, encoding="utf-8")
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
    candidates = build_candidates(snapshot, config)
    if candidates.empty:
        raise RuntimeError("No candidates passed the weekly picker filters")

    news_by_code = search_recent_news(candidates, config)
    report = render_report(candidates, news_by_code)
    report_path = save_report(report)
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
