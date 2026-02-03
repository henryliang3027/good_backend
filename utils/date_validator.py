import re
from datetime import datetime


class DateValidator:
    """台灣超市常見有效日期格式的驗證與提取工具"""

    # 民國年起始年份
    MINGUO_BASE_YEAR = 1911

    # 英文月份縮寫對照表
    MONTH_ABBR = {
        "JAN": 1,
        "FEB": 2,
        "MAR": 3,
        "APR": 4,
        "MAY": 5,
        "JUN": 6,
        "JUL": 7,
        "AUG": 8,
        "SEP": 9,
        "OCT": 10,
        "NOV": 11,
        "DEC": 12,
    }

    @classmethod
    def _parse_month_abbr(cls, month_str: str) -> int:
        """將英文月份縮寫轉換為數字"""
        return cls.MONTH_ABBR.get(month_str.upper(), 0)

    @staticmethod
    def validate_date(year: int, month: int, day: int) -> bool:
        """簡單驗證日期的有效性"""
        if month < 1 or month > 12:
            return False
        if day < 1:
            return False

        # 每月天數
        month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        # 閏年處理
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            month_days[1] = 29

        # 檢查'日'是否超過該月最大天數
        if day > month_days[month - 1]:
            return False

        return True

    @staticmethod
    def validate_expiry_date(year: int, month: int, day: int) -> bool:
        """驗證日期是否為未來日期（未過期）"""
        try:
            exp_date = datetime(year, month, day)
            current_date = datetime.now()
            return exp_date >= current_date
        except ValueError:
            return False

    @classmethod
    def _build_result(cls, year: int, month: int, day: int) -> dict:
        """建立成功的回傳結果"""
        return {
            "count": 1,
            "date": {
                "year": year,
                "month": month,
                "day": day,
            },
        }

    @staticmethod
    def _no_match_result() -> dict:
        """建立無匹配的回傳結果"""
        return {"count": 0, "date": None}

    @classmethod
    def extract_date(cls, text: str) -> dict:
        """
        從文字中提取日期資訊，支援台灣超市常見的有效日期格式。

        支援格式:
        - 英文月份: DD MMM YY, DD MMM YYYY, YYYY MMM DD, MMM YYYY, MMM DD YY, MMM DD YYYY
        - 民國年格式: YYY/MM/DD, YYY-MM-DD, YYY.MM.DD, YYY MM DD, YYYMMDD
        - 西元年月日: YYYY/MM/DD, YYYY-MM-DD, YYYY.MM.DD, YYYY MM DD, YYYYMMDD
        - 西元日月年: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY, DD MM YYYY, DDMMYYYY
        - 兩位數年份: YY/MM/DD, YY-MM-DD, YY.MM.DD, YY MM DD
        - 年月格式: YYYY/MM, YYYY-MM (day 預設為 1)
        - 月年格式: MM/YYYY, MM-YYYY (day 預設為 1)
        - 月日格式: MM/DD, MM-DD (year 使用當前年份)

        分隔符: '.' '-' '/' 空格

        Returns:
            dict with count (1 if found, 0 if not) and date (year, month, day or None)
        """
        text = text.strip()

        # 英文月份格式 (三個部分): DD MMM YY, DD MMM YYYY, YYYY MMM DD
        # 分隔符: '.', '-', '/', 空格
        pattern_eng_month_3 = (
            r"(\d{1,4})\s*[/\-\.\s]\s*([A-Za-z]{3})\s*[/\-\.\s]\s*(\d{2,4})"
        )
        match = re.search(pattern_eng_month_3, text, re.IGNORECASE)
        if match:
            part1 = int(match.group(1))
            month_str = match.group(2)
            part3 = int(match.group(3))
            month = cls._parse_month_abbr(month_str)

            if month > 0:
                if part1 > 31:  # YYYY MMM DD
                    year = part1
                    day = part3
                else:  # DD MMM YY 或 DD MMM YYYY
                    day = part1
                    year = part3
                    if year < 100:  # 兩位數年份，假設為 2000 年代
                        year += 2000

                if cls.validate_date(year, month, day):
                    return cls._build_result(year, month, day)

        # 英文月份格式 (兩個部分): MMM YYYY, MMM YY (無日期，預設 day=1)
        pattern_eng_month_2 = r"^([A-Za-z]{3})\s*[/\-\.\s]\s*(\d{2,4})$"
        match = re.search(pattern_eng_month_2, text, re.IGNORECASE)
        if match:
            month_str = match.group(1)
            year = int(match.group(2))
            month = cls._parse_month_abbr(month_str)

            if month > 0:
                if year < 100:
                    year += 2000
                day = 1
                if cls.validate_date(year, month, day):
                    return cls._build_result(year, month, day)

        # MMM DD YY, MMM DD YYYY 格式
        pattern_mmmddyy = (
            r"^([A-Za-z]{3})\s*[/\-\.\s]\s*(\d{1,2})\s*[/\-\.\s]\s*(\d{2,4})$"
        )
        match = re.search(pattern_mmmddyy, text, re.IGNORECASE)
        if match:
            month_str = match.group(1)
            day = int(match.group(2))
            year = int(match.group(3))
            month = cls._parse_month_abbr(month_str)

            if month > 0:
                if year < 100:
                    year += 2000
                if cls.validate_date(year, month, day):
                    return cls._build_result(year, month, day)

        # YY MM DD 格式 (2位數年份)
        pattern_yymmdd = r"^(\d{2})\s*[/\-\.\s]\s*(\d{1,2})\s*[/\-\.\s]\s*(\d{1,2})$"
        match = re.search(pattern_yymmdd, text)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            if year < 100:
                year += 2000
            if cls.validate_date(year, month, day):
                return cls._build_result(year, month, day)

        # YYYY MM 格式 (無日期，預設 day=1)
        pattern_yyyymm = r"^(\d{4})\s*[/\-\.\s]\s*(\d{1,2})$"
        match = re.search(pattern_yyyymm, text)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            day = 1
            if cls.validate_date(year, month, day):
                return cls._build_result(year, month, day)

        # MM YYYY 格式 (無日期，預設 day=1)
        pattern_mmyyyy = r"^(\d{1,2})\s*[/\-\.\s]\s*(\d{4})$"
        match = re.search(pattern_mmyyyy, text)
        if match:
            month = int(match.group(1))
            year = int(match.group(2))
            day = 1
            if cls.validate_date(year, month, day):
                return cls._build_result(year, month, day)

        # MM DD 格式 (無年份，使用當前年份)
        pattern_mmdd = r"^(\d{1,2})\s*[/\-\.\s]\s*(\d{1,2})$"
        match = re.search(pattern_mmdd, text)
        if match:
            month = int(match.group(1))
            day = int(match.group(2))
            year = datetime.now().year
            if cls.validate_date(year, month, day):
                return cls._build_result(year, month, day)

        # 民國年格式 (3位數年份): YYY-MM-DD, YYY/MM/DD, YYY.MM.DD, YYY MM DD
        pattern_minguo = r"^(\d{3})\s*[/\-\.\s]\s*(\d{1,2})\s*[/\-\.\s]\s*(\d{1,2})$"
        match = re.search(pattern_minguo, text)
        if match:
            minguo_year = int(match.group(1))
            if 1 <= minguo_year <= 200:
                year = minguo_year + cls.MINGUO_BASE_YEAR
                month = int(match.group(2))
                day = int(match.group(3))
                if cls.validate_date(year, month, day):
                    return cls._build_result(year, month, day)
                return cls._no_match_result()

        # 西元年格式:
        # YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD, YYYY MM DD 或
        # DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY, DD MM YYYY
        pattern_separated = r"(\d{1,4})\s*[/\-\.\s]\s*(\d{1,2})\s*[/\-\.\s]\s*(\d{1,4})"
        match = re.search(pattern_separated, text)
        if match:
            part1 = int(match.group(1))
            part2 = int(match.group(2))
            part3 = int(match.group(3))

            if part1 > 31:  # YYYY/MM/DD
                year, month, day = part1, part2, part3
            elif part3 > 31:  # DD/MM/YYYY
                year, month, day = part3, part2, part1
            else:  # 預設為年月日順序
                year, month, day = part1, part2, part3

            if cls.validate_date(year, month, day):
                return cls._build_result(year, month, day)
            return cls._no_match_result()

        # 無分隔符格式: YYYYMMDD, DDMMYYYY, 或 YYYMMDD (民國年)
        pattern_no_sep = r"(\d{7,8})"
        match = re.search(pattern_no_sep, text)
        if match:
            date_str = match.group(1)

            if len(date_str) == 8:
                first_four = int(date_str[:4])

                # 判斷是 YYYYMMDD 還是 DDMMYYYY
                if first_four > 1231:  # 前四位 > 1231，必定是 YYYYMMDD
                    year = first_four
                    month = int(date_str[4:6])
                    day = int(date_str[6:8])
                else:
                    # 嘗試 DDMMYYYY 格式
                    day = int(date_str[:2])
                    month = int(date_str[2:4])
                    year = int(date_str[4:8])

            else:  # YYYMMDD (民國年)
                year = int(date_str[:3]) + cls.MINGUO_BASE_YEAR
                month = int(date_str[3:5])
                day = int(date_str[5:7])

            print("Extracting date from date_str:", year, month, day)

            if cls.validate_date(year, month, day):
                return cls._build_result(year, month, day)
            return cls._no_match_result()

        # 西元日期格式有部分缺損: YYYY-MMDD
        pattern_partial = r"(\d{4})\s*[/\-\.:\s]?\s*(\d{1,2})\s*[/\-\.:\s]?\s*(\d{1,2})"
        match = re.search(pattern_partial, text)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            if cls.validate_date(year, month, day):
                return cls._build_result(year, month, day)
            return cls._no_match_result()

        return cls._no_match_result()

    @classmethod
    def extract_expiry_date(cls, text: str) -> dict:
        result = cls.extract_date(text)
        count = result["count"]
        date = result["date"]
        if count == 0:
            return result
        else:
            return {
                "count": 1,
                "date": {
                    "production": None,
                    "expiration": date,
                },
            }

    @classmethod
    def extract_multiple_dates(cls, text: str) -> dict:
        """
        從合併的文字中提取製造日期和有效日期。

        支援格式:
        - .PD 後跟隨製造日期 (Production Date)
        - .BB 後跟隨有效日期 (Best Before)

        Args:
            text: OCR 辨識結果合併後的字串，例如:
                  '.F25226B 04:49 .PD: 14 / 08/2025 .BB: 14 / 08/2026'

        Returns:
            dict with count and date containing production and expiration dates
        """
        production_date = None
        expiration_date = None
        text_upper = text.upper()

        # 找 .PD 製造日期
        pd_idx = text_upper.find(".PD")
        if pd_idx != -1:
            # 從 .PD 後面開始提取日期
            after_pd = text[pd_idx + 3 :]
            result = cls.extract_date(after_pd)
            if result["count"] == 1:
                production_date = result["date"]

        # 找 .BB 有效日期
        bb_idx = text_upper.find(".BB")
        if bb_idx != -1:
            # 從 .BB 後面開始提取日期
            after_bb = text[bb_idx + 3 :]
            result = cls.extract_date(after_bb)
            if result["count"] == 1:
                expiration_date = result["date"]

        # 根據找到的日期數量回傳結果
        if production_date and expiration_date:
            return {
                "count": 2,
                "date": {
                    "production": production_date,
                    "expiration": expiration_date,
                },
            }
        elif production_date:
            return {
                "count": 1,
                "date": {
                    "production": production_date,
                    "expiration": None,
                },
            }
        elif expiration_date:
            return {
                "count": 1,
                "date": {
                    "production": None,
                    "expiration": expiration_date,
                },
            }
        else:
            return {"count": 0, "date": None}
