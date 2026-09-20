# ============================================================
# NSE TECHNICAL BREAKOUT SCANNER - VERSION 3
#
# PURPOSE:
# Find ENTRY-READY breakout stocks instead of simply finding
# stocks that have already broken out.
#
# DAILY CHART:
#   - Breakout
#   - Retest
#   - Entry zone
#   - Stop loss
#   - Targets
#
# WEEKLY CHART:
#   - Trend confirmation
#   - EMA20 / EMA50
#   - Weekly RSI
#
# OUTPUT:
#   technicalBreakoutstocks.csv
#
# TELEGRAM:
#   Reads BOT_TOKEN and CHAT_ID from telegram_config.txt
#
# ============================================================

import os
import time
import warnings
import requests
import numpy as np
import pandas as pd
import yfinance as yf

from ta.trend import (
    EMAIndicator,
    MACD,
    ADXIndicator
)

from ta.momentum import (
    RSIIndicator,
    ROCIndicator
)

from ta.volatility import (
    AverageTrueRange,
    BollingerBands
)

from ta.volume import (
    OnBalanceVolumeIndicator,
    ChaikinMoneyFlowIndicator
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "symbols.csv"
OUTPUT_FILE = "technicalBreakoutstocks.csv"

PERIOD = "5y"
INTERVAL = "1d"

BATCH_SIZE = 100


# ============================================================
# BREAKOUT CONFIGURATION
# ============================================================

# Breakout must close at least 0.5% above resistance
BREAKOUT_BUFFER = 0.005

# Maximum distance from breakout level for entry
MAX_ENTRY_DISTANCE = 0.03       # 3%

# Beyond this level we consider stock too extended
MAX_EXTENDED_DISTANCE = 0.06     # 6%

# How many days back to search for previous breakout
RECENT_BREAKOUT_DAYS = 10

# Retest tolerance
RETEST_TOLERANCE = 0.02          # 2%

# Breakout lookbacks
BREAKOUT_LOOKBACK_20 = 20
BREAKOUT_LOOKBACK_50 = 50


# ============================================================
# VOLUME
# ============================================================

MIN_REL_VOLUME_BREAKOUT = 1.5
MAX_REL_VOLUME_RETEST = 1.5


# ============================================================
# TECHNICAL SCORE
# ============================================================

MIN_SCORE = 70
MAX_RESULTS = 10


# ============================================================
# RISK MANAGEMENT
# ============================================================

MIN_RR = 2.0

ATR_STOP_MULTIPLIER = 1.5

# Don't accept extremely wide stops
MAX_RISK_PERCENT = 10.0


# ============================================================
# WEEKLY TREND FILTER
# ============================================================

WEEKLY_EMA_FAST = 20
WEEKLY_EMA_SLOW = 50

WEEKLY_RSI_MIN = 50


# ============================================================
# TELEGRAM
# ============================================================

TELEGRAM_CONFIG_FILE = "telegram_config.txt"


def load_telegram_config():

    config = {}

    try:

        with open(
            TELEGRAM_CONFIG_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                if line.startswith("#"):
                    continue

                if "=" not in line:
                    continue

                key, value = line.split(
                    "=",
                    1
                )

                config[key.strip()] = value.strip()

    except FileNotFoundError:

        print(
            f"WARNING: {TELEGRAM_CONFIG_FILE} "
            f"not found."
        )

    return config


TELEGRAM_CONFIG = load_telegram_config()

TELEGRAM_BOT_TOKEN = TELEGRAM_CONFIG.get(
    "BOT_TOKEN",
    ""
)

TELEGRAM_CHAT_ID = TELEGRAM_CONFIG.get(
    "CHAT_ID",
    ""
)


# ============================================================
# LOAD SYMBOLS
# ============================================================

def load_symbols():

    df = pd.read_csv(
        INPUT_FILE,
        header=None
    )

    symbols = (
        df.iloc[:, 0]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    symbols = symbols[
        ~symbols.isin([
            "SYMBOL",
            "TICKER",
            "SYMBOLS"
        ])
    ]

    symbols = symbols[
        symbols.str.endswith(".NS")
    ]

    symbols = (
        symbols
        .drop_duplicates()
        .tolist()
    )

    return symbols


# ============================================================
# DOWNLOAD DAILY DATA
# ============================================================

def download_market_data(symbols):

    all_data = {}

    total = len(symbols)

    print()
    print("=" * 70)
    print(
        f"Downloading daily data for {total} NSE stocks"
    )
    print(
        f"Batch size: {BATCH_SIZE}"
    )
    print("=" * 70)

    for start in range(
        0,
        total,
        BATCH_SIZE
    ):

        batch = symbols[
            start:start + BATCH_SIZE
        ]

        batch_number = (
            start // BATCH_SIZE
        ) + 1

        total_batches = int(
            np.ceil(
                total / BATCH_SIZE
            )
        )

        print(
            f"\nBatch "
            f"{batch_number}/"
            f"{total_batches} "
            f"→ {len(batch)} stocks"
        )

        try:

            data = yf.download(
                tickers=batch,
                period=PERIOD,
                interval=INTERVAL,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False
            )

            if (
                data is None
                or data.empty
            ):
                print(
                    "  No data returned."
                )
                continue

            if isinstance(
                data.columns,
                pd.MultiIndex
            ):

                level0 = (
                    data.columns
                    .get_level_values(0)
                )

                level1 = (
                    data.columns
                    .get_level_values(1)
                )

                for symbol in batch:

                    try:

                        if symbol in level0:

                            stock_df = (
                                data[symbol]
                                .copy()
                            )

                        elif symbol in level1:

                            stock_df = (
                                data.xs(
                                    symbol,
                                    axis=1,
                                    level=1
                                )
                                .copy()
                            )

                        else:

                            continue

                        if not stock_df.empty:

                            all_data[
                                symbol
                            ] = stock_df

                    except Exception:

                        continue

            else:

                if len(batch) == 1:

                    symbol = batch[0]

                    if not data.empty:

                        all_data[
                            symbol
                        ] = data.copy()

        except Exception as e:

            print(
                f"  Download error: "
                f"{str(e)[:120]}"
            )

        print(
            f"  Successfully downloaded: "
            f"{len(all_data)} total stocks"
        )

        time.sleep(0.5)

    print()
    print("=" * 70)
    print(
        f"Total stocks downloaded: "
        f"{len(all_data)}"
    )
    print("=" * 70)

    return all_data


# ============================================================
# DAILY INDICATORS
# ============================================================

def calculate_daily_indicators(df):

    df = df.copy()

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for col in required_columns:

        if col not in df.columns:
            return None

    df = df.dropna(
        subset=required_columns
    )

    if len(df) < 300:
        return None

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["EMA20"] = EMAIndicator(
        df["Close"],
        window=20
    ).ema_indicator()

    df["EMA50"] = EMAIndicator(
        df["Close"],
        window=50
    ).ema_indicator()

    df["EMA100"] = EMAIndicator(
        df["Close"],
        window=100
    ).ema_indicator()

    df["EMA200"] = EMAIndicator(
        df["Close"],
        window=200
    ).ema_indicator()

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["RSI14"] = RSIIndicator(
        df["Close"],
        window=14
    ).rsi()

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    macd = MACD(
        df["Close"]
    )

    df["MACD"] = macd.macd()

    df["MACD_SIGNAL"] = (
        macd.macd_signal()
    )

    df["MACD_HIST"] = (
        macd.macd_diff()
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    atr = AverageTrueRange(
        df["High"],
        df["Low"],
        df["Close"],
        window=14
    )

    df["ATR14"] = (
        atr.average_true_range()
    )

    df["ATR_PERCENT"] = (
        df["ATR14"]
        / df["Close"]
        * 100
    )

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    adx = ADXIndicator(
        df["High"],
        df["Low"],
        df["Close"],
        window=14
    )

    df["ADX14"] = adx.adx()

    df["DI_PLUS"] = adx.adx_pos()

    df["DI_MINUS"] = adx.adx_neg()

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    df["ROC12"] = ROCIndicator(
        df["Close"],
        window=12
    ).roc()

    # --------------------------------------------------------
    # Bollinger Bands
    # --------------------------------------------------------

    bb = BollingerBands(
        df["Close"],
        window=20,
        window_dev=2
    )

    df["BB_HIGH"] = (
        bb.bollinger_hband()
    )

    df["BB_LOW"] = (
        bb.bollinger_lband()
    )

    df["BB_WIDTH"] = (
        bb.bollinger_wband()
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    df["VOLUME_SMA20"] = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    df["VOLUME_SMA50"] = (
        df["Volume"]
        .rolling(50)
        .mean()
    )

    df["REL_VOLUME"] = (
        df["Volume"]
        / df["VOLUME_SMA20"]
    )

    # --------------------------------------------------------
    # OBV
    # --------------------------------------------------------

    obv = OnBalanceVolumeIndicator(
        df["Close"],
        df["Volume"]
    )

    df["OBV"] = (
        obv.on_balance_volume()
    )

    df["OBV_SMA20"] = (
        df["OBV"]
        .rolling(20)
        .mean()
    )

    # --------------------------------------------------------
    # CMF
    # --------------------------------------------------------

    cmf = ChaikinMoneyFlowIndicator(
        df["High"],
        df["Low"],
        df["Close"],
        df["Volume"],
        window=20
    )

    df["CMF20"] = (
        cmf.chaikin_money_flow()
    )

    # --------------------------------------------------------
    # Resistance
    #
    # IMPORTANT:
    # SHIFT(1) prevents today's candle from becoming
    # its own resistance.
    # --------------------------------------------------------

    df["RESISTANCE20"] = (
        df["High"]
        .rolling(
            BREAKOUT_LOOKBACK_20
        )
        .max()
        .shift(1)
    )

    df["RESISTANCE50"] = (
        df["High"]
        .rolling(
            BREAKOUT_LOOKBACK_50
        )
        .max()
        .shift(1)
    )

    # --------------------------------------------------------
    # 52 WEEK HIGH / LOW
    # --------------------------------------------------------

    df["HIGH_52W"] = (
        df["High"]
        .rolling(
            252,
            min_periods=200
        )
        .max()
    )

    df["LOW_52W"] = (
        df["Low"]
        .rolling(
            252,
            min_periods=200
        )
        .min()
    )

    # --------------------------------------------------------
    # Candle
    # --------------------------------------------------------

    df["CANDLE_RANGE"] = (
        df["High"] -
        df["Low"]
    )

    df["CANDLE_BODY"] = (
        abs(
            df["Close"] -
            df["Open"]
        )
    )

    df["BODY_PERCENT"] = np.where(
        df["CANDLE_RANGE"] > 0,

        (
            df["CANDLE_BODY"]
            / df["CANDLE_RANGE"]
            * 100
        ),

        0
    )

    df["BULLISH_CANDLE"] = (
        df["Close"] >
        df["Open"]
    )

    return df


# ============================================================
# WEEKLY INDICATORS
# ============================================================

def calculate_weekly_indicators(daily_df):

    weekly = daily_df.resample(
        "W-FRI"
    ).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })

    weekly = weekly.dropna()

    if len(weekly) < 60:
        return None

    weekly["EMA20_W"] = (
        EMAIndicator(
            weekly["Close"],
            window=WEEKLY_EMA_FAST
        )
        .ema_indicator()
    )

    weekly["EMA50_W"] = (
        EMAIndicator(
            weekly["Close"],
            window=WEEKLY_EMA_SLOW
        )
        .ema_indicator()
    )

    weekly["RSI_W"] = (
        RSIIndicator(
            weekly["Close"],
            window=14
        )
        .rsi()
    )

    return weekly


# ============================================================
# WEEKLY TREND
# ============================================================

def weekly_trend_confirmation(
    weekly
):

    if weekly is None:
        return False

    current = weekly.iloc[-1]

    close = current["Close"]

    ema20 = current["EMA20_W"]

    ema50 = current["EMA50_W"]

    rsi = current["RSI_W"]

    if any(
        pd.isna(x)
        for x in [
            close,
            ema20,
            ema50,
            rsi
        ]
    ):
        return False

    conditions = [
        close > ema20,
        ema20 > ema50,
        rsi >= WEEKLY_RSI_MIN
    ]

    return all(conditions)


# ============================================================
# FRESH BREAKOUT DETECTION
# ============================================================

def detect_fresh_breakout(df):

    current = df.iloc[-1]

    close = current["Close"]

    resistance20 = (
        current["RESISTANCE20"]
    )

    resistance50 = (
        current["RESISTANCE50"]
    )

    rel_volume = (
        current["REL_VOLUME"]
    )

    bullish = (
        current["BULLISH_CANDLE"]
    )

    body = (
        current["BODY_PERCENT"]
    )

    # --------------------------------------------------------
    # 50-day breakout
    # --------------------------------------------------------

    breakout50 = (

        pd.notna(resistance50)

        and close >
        resistance50 *
        (1 + BREAKOUT_BUFFER)

        and rel_volume >=
        MIN_REL_VOLUME_BREAKOUT

        and bullish

        and body >= 40
    )

    if breakout50:

        return (
            "FRESH_BREAKOUT_50",
            float(resistance50)
        )

    # --------------------------------------------------------
    # 20-day breakout
    # --------------------------------------------------------

    breakout20 = (

        pd.notna(resistance20)

        and close >
        resistance20 *
        (1 + BREAKOUT_BUFFER)

        and rel_volume >=
        MIN_REL_VOLUME_BREAKOUT

        and bullish

        and body >= 40
    )

    if breakout20:

        return (
            "FRESH_BREAKOUT_20",
            float(resistance20)
        )

    return None, None


# ============================================================
# PREVIOUS BREAKOUT / RETEST
# ============================================================

def detect_retest(df):

    if len(df) < (
        RECENT_BREAKOUT_DAYS + 60
    ):
        return None, None

    current = df.iloc[-1]

    recent = df.iloc[
        -(RECENT_BREAKOUT_DAYS + 1):-1
    ]

    candidates = []

    for idx, row in recent.iterrows():

        resistance50 = (
            row["RESISTANCE50"]
        )

        resistance20 = (
            row["RESISTANCE20"]
        )

        close = row["Close"]

        rel_volume = (
            row["REL_VOLUME"]
        )

        bullish = (
            row["BULLISH_CANDLE"]
        )

        # 50-day breakout
        if (
            pd.notna(resistance50)
            and close >
            resistance50 *
            (1 + BREAKOUT_BUFFER)
            and rel_volume >=
            MIN_REL_VOLUME_BREAKOUT
            and bullish
        ):

            candidates.append(
                (
                    idx,
                    float(resistance50),
                    "50"
                )
            )

        # 20-day breakout
        elif (
            pd.notna(resistance20)
            and close >
            resistance20 *
            (1 + BREAKOUT_BUFFER)
            and rel_volume >=
            MIN_REL_VOLUME_BREAKOUT
            and bullish
        ):

            candidates.append(
                (
                    idx,
                    float(resistance20),
                    "20"
                )
            )

    if not candidates:

        return None, None

    # Most recent breakout
    breakout_date, breakout_level, breakout_type = (
        candidates[-1]
    )

    current_close = (
        current["Close"]
    )

    current_low = (
        current["Low"]
    )

    current_volume = (
        current["REL_VOLUME"]
    )

    # --------------------------------------------------------
    # Distance from breakout
    # --------------------------------------------------------

    distance = (
        current_close -
        breakout_level
    ) / breakout_level

    # Too far above breakout
    if distance > MAX_EXTENDED_DISTANCE:

        return (
            "EXTENDED",
            breakout_level
        )

    # --------------------------------------------------------
    # Retest condition
    #
    # Price must come close to breakout level.
    # --------------------------------------------------------

    low_distance = abs(
        current_low -
        breakout_level
    ) / breakout_level

    near_level = (
        low_distance <=
        RETEST_TOLERANCE
    )

    reclaim = (
        current_close >=
        breakout_level
    )

    bullish = (
        current["Close"] >
        current["Open"]
    )

    lower_volume = (
        current_volume <
        MAX_REL_VOLUME_RETEST
    )

    if (
        near_level
        and reclaim
        and bullish
        and lower_volume
    ):

        return (
            "RETEST_ENTRY",
            breakout_level
        )

    # --------------------------------------------------------
    # Price still reasonably close to breakout
    # --------------------------------------------------------

    if (
        distance >= 0
        and distance <=
        MAX_ENTRY_DISTANCE
    ):

        return (
            "BREAKOUT_ENTRY",
            breakout_level
        )

    # --------------------------------------------------------
    # If price is below breakout but very close,
    # wait for reclaim.
    # --------------------------------------------------------

    if (
        distance < 0
        and abs(distance) <=
        RETEST_TOLERANCE
    ):

        return (
            "WAIT_RECLAIM",
            breakout_level
        )

    return None, None


# ============================================================
# ENTRY QUALITY
# ============================================================

def entry_quality(
    df,
    breakout_level,
    setup
):

    current = df.iloc[-1]

    close = float(
        current["Close"]
    )

    distance = (
        close -
        breakout_level
    ) / breakout_level

    # --------------------------------------------------------
    # Extended stock
    # --------------------------------------------------------

    if distance > MAX_EXTENDED_DISTANCE:

        return "EXTENDED"

    # --------------------------------------------------------
    # Fresh breakout
    # --------------------------------------------------------

    if setup.startswith(
        "FRESH_BREAKOUT"
    ):

        if distance <= MAX_ENTRY_DISTANCE:

            return "ENTRY_READY"

        return "EXTENDED"

    # --------------------------------------------------------
    # Retest
    # --------------------------------------------------------

    if setup == "RETEST_ENTRY":

        return "ENTRY_READY"

    if setup == "BREAKOUT_ENTRY":

        return "ENTRY_READY"

    return "WAIT"


# ============================================================
# TECHNICAL SCORE
# ============================================================

def calculate_score(
    df,
    weekly,
    setup
):

    current = df.iloc[-1]

    score = 0

    # --------------------------------------------------------
    # DAILY TREND
    # --------------------------------------------------------

    if current["Close"] > current["EMA20"]:
        score += 5

    if current["Close"] > current["EMA50"]:
        score += 5

    if current["Close"] > current["EMA200"]:
        score += 10

    if current["EMA20"] > current["EMA50"]:
        score += 5

    if current["EMA50"] > current["EMA200"]:
        score += 5

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if (
        55 <=
        current["RSI14"] <=
        75
    ):
        score += 8

    # Avoid extremely overheated RSI
    if current["RSI14"] > 80:
        score -= 5

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    if (
        current["MACD"] >
        current["MACD_SIGNAL"]
    ):
        score += 7

    if current["MACD_HIST"] > 0:
        score += 5

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    if current["ADX14"] >= 20:
        score += 5

    if (
        current["DI_PLUS"] >
        current["DI_MINUS"]
    ):
        score += 5

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    if (
        current["REL_VOLUME"] >=
        MIN_REL_VOLUME_BREAKOUT
    ):
        score += 8

    # --------------------------------------------------------
    # OBV
    # --------------------------------------------------------

    if (
        current["OBV"] >
        current["OBV_SMA20"]
    ):
        score += 4

    # --------------------------------------------------------
    # CMF
    # --------------------------------------------------------

    if current["CMF20"] > 0:
        score += 4

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    if current["ROC12"] > 0:
        score += 4

    # --------------------------------------------------------
    # WEEKLY TREND
    # --------------------------------------------------------

    if weekly is not None:

        w = weekly.iloc[-1]

        if (
            w["Close"] >
            w["EMA20_W"]
        ):
            score += 5

        if (
            w["EMA20_W"] >
            w["EMA50_W"]
        ):
            score += 5

        if w["RSI_W"] >= 50:
            score += 5

    # --------------------------------------------------------
    # SETUP BONUS
    # --------------------------------------------------------

    if setup == "RETEST_ENTRY":

        score += 8

    elif setup == "FRESH_BREAKOUT_50":

        score += 6

    elif setup == "FRESH_BREAKOUT_20":

        score += 4

    elif setup == "BREAKOUT_ENTRY":

        score += 5

    return max(
        0,
        min(score, 100)
    )


# ============================================================
# TRADE LEVELS
# ============================================================

def calculate_trade_levels(
    df,
    breakout_level
):

    current = df.iloc[-1]

    entry = float(
        current["Close"]
    )

    atr = float(
        current["ATR14"]
    )

    if atr <= 0:
        return None

    # --------------------------------------------------------
    # Recent swing low
    # --------------------------------------------------------

    swing_low = float(
        df["Low"]
        .tail(10)
        .min()
    )

    # --------------------------------------------------------
    # ATR stop
    # --------------------------------------------------------

    atr_stop = (
        entry -
        ATR_STOP_MULTIPLIER *
        atr
    )

    # --------------------------------------------------------
    # Structural stop
    #
    # Use the lower of swing low and ATR stop so that
    # normal volatility doesn't trigger the stop too easily.
    # --------------------------------------------------------

    stop = min(
        swing_low,
        atr_stop
    )

    if stop >= entry:

        stop = (
            entry -
            ATR_STOP_MULTIPLIER *
            atr
        )

    risk = (
        entry -
        stop
    )

    if risk <= 0:
        return None

    risk_percent = (
        risk /
        entry *
        100
    )

    # Reject very wide stop
    if risk_percent > MAX_RISK_PERCENT:

        return None

    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    target1 = (
        entry +
        risk * 2
    )

    target2 = (
        entry +
        risk * 3
    )

    target3 = (
        entry +
        risk * 4
    )

    rr1 = (
        target1 -
        entry
    ) / risk

    # --------------------------------------------------------
    # Breakout distance
    # --------------------------------------------------------

    distance_from_breakout = (
        (
            entry -
            breakout_level
        )
        / breakout_level
        * 100
    )

    return {

        "Entry": entry,

        "Stop_Loss": stop,

        "Target1": target1,

        "Target2": target2,

        "Target3": target3,

        "Risk_Per_Share": risk,

        "Risk_Percent": risk_percent,

        "RR": rr1,

        "Distance_From_Breakout_%":
            distance_from_breakout,

        "Breakout_Level":
            breakout_level
    }


# ============================================================
# ANALYSE ONE STOCK
# ============================================================

def analyse_stock(
    symbol,
    raw_df
):

    try:

        # ----------------------------------------------------
        # Daily indicators
        # ----------------------------------------------------

        df = calculate_daily_indicators(
            raw_df
        )

        if df is None:
            return None

        # ----------------------------------------------------
        # Weekly indicators
        # ----------------------------------------------------

        weekly = (
            calculate_weekly_indicators(
                df
            )
        )

        if weekly is None:
            return None

        # ----------------------------------------------------
        # Weekly trend confirmation
        # ----------------------------------------------------

        weekly_confirmed = (
            weekly_trend_confirmation(
                weekly
            )
        )

        # We require weekly trend confirmation
        if not weekly_confirmed:

            return None

        # ----------------------------------------------------
        # Detect today's fresh breakout
        # ----------------------------------------------------

        setup, breakout_level = (
            detect_fresh_breakout(df)
        )

        # ----------------------------------------------------
        # If no fresh breakout,
        # look for recent retest
        # ----------------------------------------------------

        if setup is None:

            setup, breakout_level = (
                detect_retest(df)
            )

        if setup is None:

            return None

        # Extended stocks should not be returned
        if setup == "EXTENDED":

            return None

        # ----------------------------------------------------
        # Entry quality
        # ----------------------------------------------------

        quality = entry_quality(
            df,
            breakout_level,
            setup
        )

        if quality != "ENTRY_READY":

            return None

        # ----------------------------------------------------
        # Score
        # ----------------------------------------------------

        score = calculate_score(
            df,
            weekly,
            setup
        )

        if score < MIN_SCORE:

            return None

        # ----------------------------------------------------
        # Trade levels
        # ----------------------------------------------------

        trade = calculate_trade_levels(
            df,
            breakout_level
        )

        if trade is None:

            return None

        if trade["RR"] < MIN_RR:

            return None

        current = df.iloc[-1]

        weekly_current = (
            weekly.iloc[-1]
        )

        # ----------------------------------------------------
        # 52-week correction
        # ----------------------------------------------------

        if pd.notna(
            current["HIGH_52W"]
        ):

            correction = (

                (
                    current["HIGH_52W"]
                    -
                    current["Close"]
                )
                /
                current["HIGH_52W"]
                *
                100
            )

        else:

            correction = np.nan

        # ----------------------------------------------------
        # Date
        # ----------------------------------------------------

        last_date = df.index[-1]

        if hasattr(
            last_date,
            "date"
        ):

            last_date = (
                last_date.date()
            )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        result = {

            "Date":
                last_date,

            "Symbol":
                symbol,

            "Setup":
                setup,

            "Entry_Status":
                quality,

            # ------------------------------------------------
            # Price
            # ------------------------------------------------

            "Close":
                round(
                    current["Close"],
                    2
                ),

            "Breakout_Level":
                round(
                    breakout_level,
                    2
                ),

            "Distance_From_Breakout_%":
                round(
                    trade[
                        "Distance_From_Breakout_%"
                    ],
                    2
                ),

            # ------------------------------------------------
            # Trade
            # ------------------------------------------------

            "Entry":
                round(
                    trade["Entry"],
                    2
                ),

            "Stop_Loss":
                round(
                    trade["Stop_Loss"],
                    2
                ),

            "Target1":
                round(
                    trade["Target1"],
                    2
                ),

            "Target2":
                round(
                    trade["Target2"],
                    2
                ),

            "Target3":
                round(
                    trade["Target3"],
                    2
                ),

            "Risk_Per_Share":
                round(
                    trade["Risk_Per_Share"],
                    2
                ),

            "Risk_%":
                round(
                    trade["Risk_Percent"],
                    2
                ),

            "RR":
                round(
                    trade["RR"],
                    2
                ),

            # ------------------------------------------------
            # Daily technical
            # ------------------------------------------------

            "EMA20":
                round(
                    current["EMA20"],
                    2
                ),

            "EMA50":
                round(
                    current["EMA50"],
                    2
                ),

            "EMA200":
                round(
                    current["EMA200"],
                    2
                ),

            "RSI14":
                round(
                    current["RSI14"],
                    2
                ),

            "ADX14":
                round(
                    current["ADX14"],
                    2
                ),

            "DI_PLUS":
                round(
                    current["DI_PLUS"],
                    2
                ),

            "DI_MINUS":
                round(
                    current["DI_MINUS"],
                    2
                ),

            "MACD_HIST":
                round(
                    current["MACD_HIST"],
                    2
                ),

            "ATR_%":
                round(
                    current["ATR_PERCENT"],
                    2
                ),

            "REL_VOLUME":
                round(
                    current["REL_VOLUME"],
                    2
                ),

            "ROC12":
                round(
                    current["ROC12"],
                    2
                ),

            "CMF20":
                round(
                    current["CMF20"],
                    3
                ),

            # ------------------------------------------------
            # Weekly
            # ------------------------------------------------

            "Weekly_Close":
                round(
                    weekly_current["Close"],
                    2
                ),

            "Weekly_EMA20":
                round(
                    weekly_current["EMA20_W"],
                    2
                ),

            "Weekly_EMA50":
                round(
                    weekly_current["EMA50_W"],
                    2
                ),

            "Weekly_RSI":
                round(
                    weekly_current["RSI_W"],
                    2
                ),

            "Weekly_Trend":
                "BULLISH",

            # ------------------------------------------------
            # 52 Week
            # ------------------------------------------------

            "52W_High":
                round(
                    current["HIGH_52W"],
                    2
                ),

            "52W_Low":
                round(
                    current["LOW_52W"],
                    2
                ),

            "Correction_From_52W_High_%":
                round(
                    correction,
                    2
                ),

            # ------------------------------------------------
            # Score
            # ------------------------------------------------

            "Technical_Score":
                score
        }

        return result

    except Exception as e:

        print(
            f"Error analysing "
            f"{symbol}: "
            f"{str(e)[:120]}"
        )

        return None


# ============================================================
# TELEGRAM MESSAGE
# ============================================================

def send_telegram_message(
    message
):

    if not TELEGRAM_BOT_TOKEN:

        print(
            "Telegram BOT_TOKEN missing."
        )

        return

    if not TELEGRAM_CHAT_ID:

        print(
            "Telegram CHAT_ID missing."
        )

        return

    url = (
        "https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/"
        "sendMessage"
    )

    payload = {

        "chat_id":
            TELEGRAM_CHAT_ID,

        "text":
            message
    }

    try:

        response = requests.post(
            url,
            data=payload,
            timeout=30
        )

        response.raise_for_status()

        print(
            "Telegram message sent."
        )

    except Exception as e:

        print(
            f"Telegram error: {e}"
        )


# ============================================================
# TELEGRAM CSV UPLOAD
# ============================================================

def send_telegram_file(
    file_path
):

    if not TELEGRAM_BOT_TOKEN:
        return

    if not TELEGRAM_CHAT_ID:
        return

    url = (
        "https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/"
        "sendDocument"
    )

    try:

        with open(
            file_path,
            "rb"
        ) as file:

            files = {
                "document": file
            }

            data = {
                "chat_id":
                    TELEGRAM_CHAT_ID
            }

            response = requests.post(
                url,
                data=data,
                files=files,
                timeout=30
            )

        response.raise_for_status()

        print(
            f"{file_path} "
            "sent to Telegram."
        )

    except Exception as e:

        print(
            f"Telegram file error: "
            f"{e}"
        )


# ============================================================
# TELEGRAM SUMMARY
# ============================================================

def create_telegram_message(
    results
):

    if not results:

        return (
            "NSE ENTRY SCANNER\n\n"
            "No entry-ready stocks found "
            "today.\n\n"
            "Weekly trend + daily breakout/"
            "retest + risk/reward filters "
            "were applied."
        )

    lines = [

        "NSE ENTRY-READY BREAKOUT SCANNER",

        "",

        f"Candidates: "
        f"{len(results)}",

        ""
    ]

    for i, row in enumerate(
        results,
        start=1
    ):

        lines.append(

            f"{i}. "
            f"{row['Symbol']} | "
            f"{row['Setup']}"
        )

        lines.append(

            f"   Entry ₹"
            f"{row['Entry']:.2f} | "
            f"SL ₹"
            f"{row['Stop_Loss']:.2f}"
        )

        lines.append(

            f"   T1 ₹"
            f"{row['Target1']:.2f} | "
            f"RR "
            f"{row['RR']:.1f} | "
            f"Score "
            f"{row['Technical_Score']}"
        )

        lines.append(

            f"   Breakout "
            f"{row['Distance_From_Breakout_%']:.1f}% "
            f"away"
        )

        lines.append("")

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print()
    print("=" * 70)
    print(
        "NSE TECHNICAL BREAKOUT SCANNER - VERSION 3"
    )
    print("=" * 70)

    print()
    print(
        "Objective:"
    )

    print(
        "Find entry-ready stocks, "
        "not stocks that already rallied."
    )

    print()
    print(
        "Daily chart = breakout / entry"
    )

    print(
        "Weekly chart = trend confirmation"
    )

    print(
        f"Maximum results = {MAX_RESULTS}"
    )

    # --------------------------------------------------------
    # Load symbols
    # --------------------------------------------------------

    symbols = load_symbols()

    print()
    print(
        f"Symbols in CSV: "
        f"{len(symbols)}"
    )

    if not symbols:

        print(
            "No valid .NS symbols found."
        )

        return

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    market_data = (
        download_market_data(
            symbols
        )
    )

    # --------------------------------------------------------
    # Analyse
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "Running daily + weekly analysis..."
    )
    print("=" * 70)

    results = []

    processed = 0

    for symbol, df in (
        market_data.items()
    ):

        processed += 1

        result = analyse_stock(
            symbol,
            df
        )

        if result is not None:

            results.append(
                result
            )

        if processed % 100 == 0:

            print(
                f"Analysed "
                f"{processed}/"
                f"{len(market_data)}"
            )

    # --------------------------------------------------------
    # Dataframe
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    if not results_df.empty:

        # ----------------------------------------------------
        # Ranking:
        #
        # 1. Technical score
        # 2. Lower distance from breakout
        # 3. Better RR
        # ----------------------------------------------------

        results_df = (
            results_df
            .sort_values(
                by=[
                    "Technical_Score",
                    "Distance_From_Breakout_%",
                    "RR"
                ],
                ascending=[
                    False,
                    True,
                    False
                ]
            )
        )

        # Keep only top candidates
        results_df = (
            results_df
            .head(MAX_RESULTS)
            .reset_index(
                drop=True
            )
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 70)

    print(
        f"Entry-ready candidates: "
        f"{len(results_df)}"
    )

    print(
        f"Output file: "
        f"{OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    if not results_df.empty:

        display_columns = [

            "Symbol",

            "Setup",

            "Close",

            "Breakout_Level",

            "Distance_From_Breakout_%",

            "Entry",

            "Stop_Loss",

            "Target1",

            "Target2",

            "Risk_%",

            "RR",

            "Technical_Score",

            "Weekly_RSI"
        ]

        print()

        print(
            results_df[
                display_columns
            ].to_string(
                index=False
            )
        )

    else:

        print()
        print(
            "No entry-ready stocks "
            "matched all filters."
        )

    # --------------------------------------------------------
    # Telegram
    # --------------------------------------------------------

    telegram_results = (
        results_df
        .to_dict(
            orient="records"
        )
    )

    telegram_message = (
        create_telegram_message(
            telegram_results
        )
    )

    send_telegram_message(
        telegram_message
    )

    send_telegram_file(
        OUTPUT_FILE
    )

    # --------------------------------------------------------
    # Runtime
    # --------------------------------------------------------

    elapsed = (
        time.time() -
        start_time
    )

    print()
    print("=" * 70)

    print(
        f"Total processing time: "
        f"{elapsed / 60:.2f} minutes"
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()